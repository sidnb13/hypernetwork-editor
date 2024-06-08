import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from models.utils import EditorModelOutput

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def visualize_attn_heatmap(
    result: EditorModelOutput,
    orig_logits: torch.Tensor,
    batch: Dict,
    save_path: str | os.PathLike = None,
    show_plot: bool = False,
    tokenizer: AutoTokenizer = None,
    stopping_index: int = None,
    metadata: DictConfig = None,
    step: int = None,
):
    if save_path:
        if step is not None:
            save_path = Path(save_path) / "viz-step-{}".format(step)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(save_path) / f"viz-{timestamp}"
        show_plot = False
        os.makedirs(save_path, exist_ok=True)

        if metadata:
            OmegaConf.save(metadata, save_path / "config.yaml")

    for batch_index in range(len(next(iter(batch.values())))):
        # The tensor norm comes in an stopping_index * num_layers+1 matrix
        target_attn_mask = batch["target_attention_mask"][batch_index]
        target_input_ids = batch["target_input_ids"][batch_index]
        result_logits = result.logits[batch_index]

        using_ghost_token = False
        # Add in the ghost token to the mask and target id's
        if target_attn_mask.shape[0] == result.edit_vectors[batch_index].shape[0] - 1:
            using_ghost_token = True
            target_attn_mask = torch.cat(
                [torch.ones(1, device=target_attn_mask.device), target_attn_mask]
            )
            # also add a placeholder token id = 0 to the input_ids
            target_input_ids = torch.cat(
                [
                    torch.zeros(
                        1, device=target_input_ids.device, dtype=target_input_ids.dtype
                    ),
                    target_input_ids,
                ]
            )

        edit_tensor = result.edit_vectors[batch_index][target_attn_mask > 0].cpu()
        target_hidden = result.target_hidden_states[batch_index].cpu()

        edit_tensor[:stopping_index, :, :] = edit_tensor[
            :stopping_index, :, :
        ] / target_hidden[:stopping_index].norm(dim=2, keepdim=True)
        edit_tensor_norm = edit_tensor.norm(dim=2).flip(1)

        # is this any better??
        # attention_matrix = result['editor_attention'][batch_index].reshape(104).to("cpu").reshape(13,8).permute(1,0)

        # Detach and convert to numpy
        edit_tensor_norm = edit_tensor_norm.numpy()[:stopping_index, :]

        # Create the heatmap
        fig, ax = plt.subplots()
        heatmap = ax.imshow(edit_tensor_norm.transpose(), cmap="hot")

        # Color the heatmap according to the entry sizes
        heatmap.set_clim(vmin=np.min(0), vmax=np.max(edit_tensor_norm))
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Entry Sizes")

        # TODO: pass model config prevent hardcoding
        # Add labels to the x and y axes
        ax.set_yticks(np.arange(13))
        ax.set_xticks(np.arange(8))
        # ax.set_yticklabels(np.arange(13))
        ax.set_yticklabels(np.arange(12, -1, -1))
        ax.set_xticklabels(np.arange(8))

        # Rotate the x-axis labels
        # plt.xticks(rotation=90)
        # Add a title
        plt.title("Edit / Target Norm Heatmap")

        editing_target_tokens = target_input_ids[target_attn_mask > 0]
        if stopping_index is not None:
            editing_target_tokens = editing_target_tokens[:stopping_index]

        editing_target = tokenizer.batch_decode(
            editing_target_tokens,
            skip_special_tokens=True,
        )
        editor_input = tokenizer.batch_decode(
            batch["editor_input_ids"][batch_index][
                batch["editor_attention_mask"][batch_index] > 0
            ],
            skip_special_tokens=True,
        )
        if using_ghost_token:
            selection = target_attn_mask > 0
            select_logits = (
                result_logits[selection[1:]][:stopping_index].cpu()
                if stopping_index
                else result_logits[target_attn_mask > 0].cpu()
            )
        else:
            select_logits = (
                result_logits[target_attn_mask > 0][:stopping_index].cpu()
                if stopping_index
                else result_logits[target_attn_mask > 0].cpu()
            )
        editor_preds = torch.argmax(select_logits.softmax(-1), dim=-1)
        editor_preds = tokenizer.batch_decode(editor_preds, skip_special_tokens=True)

        # model without intervention
        orig_preds = torch.argmax(orig_logits[batch_index].softmax(-1), dim=-1)
        orig_preds = tokenizer.batch_decode(orig_preds, skip_special_tokens=True)

        if show_plot:
            print("Editing target:", editing_target)
            print("Editor input:", editor_input)
            print("Editor preds:", editor_preds)
            print("Orig preds:", orig_preds)

        if show_plot:
            plt.show()

        if not save_path:
            continue

        batch_path = save_path / f"batch_{batch_index}"
        os.makedirs(batch_path, exist_ok=True)

        plt.savefig(batch_path / "attn_heatmap.png")
        if not show_plot:
            plt.close()

        with open(batch_path / "preds.json", "w") as f:
            preds = {
                "editing_target": editing_target,
                "editor_input": editor_input,
                "editor_preds": editor_preds,
                "orig_preds": orig_preds,
            }

            json.dump(preds, f)


def get_nb_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_bytes = (
                param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            )
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_tokenizer(name_or_path, padding_side="right"):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token_id = tok.eos_token_id
    # This is very important because we take last hidden state in editor
    tok.padding_side = padding_side
    return tok


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {
        k: (v.to(rank) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()
    }
    return on_device


def concat_and_pad_ids(batch: dict, pad_token: int):
    first, second = batch["editor_input_ids"], batch["target_input_ids"]
    batch_size, _ = first.size()
    # Find the lengths in A and B
    lengths_A = torch.sum(batch["editor_attention_mask"] > 0, dim=1)
    lengths_B = torch.sum(batch["target_attention_mask"] > 0, dim=1)

    # initialize empty tensor
    max_len = max(lengths_A + lengths_B)
    result = torch.full(
        (
            batch_size,
            max_len,
        ),
        pad_token,
        device=first.device,
        dtype=first.dtype,
    )
    # Concatenate A[i] and B[i] a, assume RIGHT padding
    for i in range(batch_size):
        result[i, : lengths_A[i]] = first[i, : lengths_A[i]]
        result[i, lengths_A[i] : lengths_A[i] + lengths_B[i] :] = second[
            i, : lengths_B[i]
        ]

    return result


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = field(default_factory=list)
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def __len__(self):
        return len(self.messages)

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}: {message}{self.sep}"
                else:
                    ret += f"{role}: "
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.METAMATH:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                # For MetaMath, sep2 is used to prefix the message.
                starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
                ending_sep = self.sep if i % 2 == 0 else ""
                if message:
                    ret += role + starting_sep + message + ending_sep
                else:
                    ret += role + starting_sep
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = system_prompt + self.sep + self.sep2 * 2
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + self.sep2 + message + self.sep + self.sep2 * 2
                else:
                    ret += role + self.sep2
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append((role, message))

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca_input",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        system_template="### Instruction:\n{system_message}",
        roles=("### Input:", "### Response:"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca_no_input",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction:", "### Response:"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        sep2="</s>",
    )
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="<s>[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="hermes",
        system_template="<|im_start|>system\n{system_message}",
        system_message="""You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32002, 0],
    )
)

register_conv_template(
    Conversation(
        name="llama3",
        system_template="<|start_header_id|>system<|end_header_id|>\n{system_message}",
        roles=(
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
        ),
        sep_style=SeparatorStyle.LLAMA3,
        sep="<|eot_id|>",
        sep2="\n",
    )
)


def get_conv_template(name: str) -> Conversation:
    return conv_templates[name].copy()
