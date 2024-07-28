import copy
import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Callable, List, Literal

import datasets
import torch
import torch.distributed as dist
import transformers
from click import edit
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from helpers import COLOR_MAP, NUM2WORD, ROOT_DIR, get_tokenizer
from logger import get_logger

logger = get_logger(__name__)


class DatasetCache:
    def __init__(
        self,
        hash_keys: List[str],
        cache_dir: str | os.PathLike = None,
    ) -> None:
        self.hash_keys = hash_keys
        self.cache_dir = cache_dir or os.path.join(ROOT_DIR, "assets/data/cache")

    def __call__(self, func) -> Any:
        def wrapper(*args, **kwargs):
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            hash_config = {
                k: OmegaConf.to_object(getattr(args[0], k)) for k in self.hash_keys
            }
            signature = hashlib.md5(json.dumps(hash_config).encode("utf-8")).hexdigest()

            # Get the source code of the function as a string
            function_source = inspect.getsource(func)
            # Encode the source code using UTF-8
            encoded_source = function_source.encode("utf-8")
            # Compute the MD5 hash of the encoded source code
            signature += hashlib.md5(encoded_source).hexdigest()

            is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
            rank = 0 if is_rank0 else dist.get_rank()

            def barrier():
                return dist.barrier() if dist.is_initialized() else None

            cache_folder = os.path.join(self.cache_dir, f"{func.__name__}_{signature}")
            use_cache = os.environ.get("USE_CACHE", "true").lower() == "true"
            if use_cache:
                if os.path.exists(cache_folder):
                    barrier()
                    logger.debug(
                        f"Loading cached huggingface dataset from {cache_folder} ({rank=})"
                    )
                    return datasets.load_from_disk(cache_folder)
                elif is_rank0:
                    logger.debug(f"Processing huggingface dataset ({rank=})")
                    result = func(*args, **kwargs)
                    result.save_to_disk(cache_folder)
                    return result
            else:
                if is_rank0:
                    logger.debug(f"Processing huggingface dataset ({rank=})")
                    result = func(*args, **kwargs)
                    result.save_to_disk(cache_folder)
                    return result
                else:
                    barrier()
                    return datasets.load_from_disk(cache_folder)

        return wrapper


@DatasetCache(hash_keys=["task"])
def load_wikipedia(config: DictConfig):
    assert config.task.name == "wikipedia", "task must be 'wikipedia'"

    dataset = datasets.load_dataset("abokbot/wikipedia-first-paragraph", split="train")
    tokenizer = get_tokenizer(config.model.name_or_path)

    assert tokenizer.padding_side == "right", "padding_side must be 'right'"

    def extract_segments(texts: dict):
        segment_a = []
        segment_b = []
        segment_c = []
        segment_d = []

        for text in texts["text"]:
            # Split the text into 4 segments
            total_chars = len(text)
            a_end = min(config.task.seq_a, total_chars)
            b_end = min(a_end + config.task.seq_b, total_chars)
            c_end = min(b_end + config.task.seq_c, total_chars)

            segment_a.append(text[:a_end])
            segment_b.append(text[a_end:b_end])
            segment_c.append(text[b_end:c_end])
            segment_d.append(text[c_end:])

        return {
            "segment_a": segment_a,
            "segment_b": segment_b,
            "segment_c": segment_c,
            "segment_d": segment_d,
        }

    new_dataset = dataset.map(extract_segments, batched=True, num_proc=os.cpu_count())

    new_dataset = new_dataset.remove_columns(["text", "url", "id"])
    new_dataset = new_dataset.filter(
        lambda row: all(
            len(seg) > 0
            for seg in [
                row["segment_a"],
                row["segment_b"],
                row["segment_c"],
                row["segment_d"],
            ]
        ),
        num_proc=os.cpu_count(),
    )

    new_dataset = new_dataset.map(
        tokenize_editor_pretrain,
        batched=True,
        num_proc=os.cpu_count(),
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.model.max_length,
            "editor_token_limit": config.task.editor_token_limit,
        },
    )

    # filter empty target sequences
    limit = config.train.stop_editing_idx or 0
    new_dataset = new_dataset.filter(
        lambda row: sum(row["target_attention_mask"]) > limit, num_proc=os.cpu_count()
    )

    new_dataset.set_format(
        "torch",
        columns=[
            col
            for col in new_dataset.column_names
            if "target_" in col or "editor_" in col
        ],
    )

    return new_dataset


def tokenize_editor_pretrain(
    row_batch: dict, tokenizer=None, max_length=None, editor_token_limit=None
):
    # Concatenate segments A and D for editor inputs
    editor_text = [
        a + " " + d for a, d in zip(row_batch["segment_a"], row_batch["segment_d"])
    ]

    # Concatenate segments B and C for target inputs
    target_text = [
        b + " " + c for b, c in zip(row_batch["segment_b"], row_batch["segment_c"])
    ]

    editor_inputs = tokenizer(
        editor_text,
        max_length=max_length,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
    )

    target_inputs = tokenizer(
        target_text,
        add_special_tokens=False,
        max_length=editor_token_limit,
        padding="max_length",
        truncation=True,
    )

    return {
        **{"editor_" + k: v for k, v in editor_inputs.items()},
        **{"target_" + k: v for k, v in target_inputs.items()},
    }


def tokenize_editor_eval(
    row_batch: dict,
    tokenizer=None,
    max_length=None,
    editor_token_limit=None,
    editor_input_col=None,
    target_input_col=None,
    target_col=None,
):
    editor_inputs = tokenizer(
        row_batch[editor_input_col],
        add_special_tokens=False,
        max_length=editor_token_limit,
        truncation=True,
    )

    # Add eos token to the start of target inputs
    target_inputs_with_eos = [
        tokenizer.bos_token + text for text in row_batch[target_input_col]
    ]

    target_inputs = tokenizer(
        target_inputs_with_eos,
        max_length=max_length,
        add_special_tokens=False,
        truncation=True,
    )
    # reverse, pad, unreverse
    target_input_ids = pad(
        target_inputs.input_ids, tokenizer.pad_token_id, editor_token_limit, "left"
    )
    target_attention_mask = pad(
        target_inputs.attention_mask, 0, editor_token_limit, "left"
    )
    editor_input_ids = pad(
        editor_inputs.input_ids, tokenizer.pad_token_id, max_length, "right"
    )
    editor_attention_mask = pad(editor_inputs.attention_mask, 0, max_length, "right")
    # add eos token to the start of target outputs
    target_outputs_with_eos = [
        text + tokenizer.eos_token for text in row_batch[target_col]
    ]
    target_outputs = tokenizer(
        target_outputs_with_eos,
        max_length=max_length,
        add_special_tokens=False,
        truncation=True,
    )

    target_outputs.input_ids = pad(
        target_outputs.input_ids, tokenizer.pad_token_id, max_length, "right"
    )

    return {
        "editor_input_ids": editor_input_ids,
        "editor_attention_mask": editor_attention_mask,
        "target_input_ids": target_input_ids,
        "target_attention_mask": target_attention_mask,
        "labels": target_outputs.input_ids,
    }


def tokenize_editor(
    row_batch: dict,
    tokenizer=None,
    max_length=None,
    editor_token_limit=None,
    instruction_col=None,
    target_col=None,
):
    editor_inputs = tokenizer(
        row_batch[instruction_col],
        add_special_tokens=False,
        max_length=editor_token_limit,
        padding="max_length",
        truncation=True,
    )
    target_inputs = tokenizer(
        row_batch[target_col],
        max_length=max_length,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
    )

    return {
        **{"editor_" + k: v for k, v in editor_inputs.items()},
        **{"target_" + k: v for k, v in target_inputs.items()},
    }


def pad(x, padding_value, max_length, padding_side):
    x = [
        torch.tensor(torch.tensor(ids[::-1] if padding_side == "left" else ids))
        for ids in x
    ]
    x[0] = torch.nn.functional.pad(
        x[0],
        (0, max_length - x[0].shape[-1]),
        value=padding_value,
    )
    padded = pad_sequence(
        x,
        batch_first=True,
        padding_value=padding_value,
    )
    # unreverse if left padding
    if padding_side == "left":
        padded = padded.flip(-1)
    return padded


def tokenize_sft(
    row_batch: dict,
    tokenizer=None,
    max_length=None,
    instruction_col=None,
    target_col=None,
    padding_side="right",
):
    tokenized_instructions = tokenizer(
        row_batch[instruction_col],
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    # add bos token
    tokenized_instructions["input_ids"] = [
        [tokenizer.bos_token_id] + x for x in tokenized_instructions["input_ids"]
    ]
    tokenized_inputs = tokenizer(
        [
            i + "\n" + t
            for i, t in zip(row_batch[instruction_col], row_batch[target_col])
        ],
        max_length=max_length,
        add_special_tokens=False,
        truncation=True,
    )
    # add bos and eos tokens
    tokenized_inputs["input_ids"] = [
        [tokenizer.bos_token_id] + x + [tokenizer.eos_token_id]
        for x in tokenized_inputs["input_ids"]
    ]
    tokenized_inputs["attention_mask"] = [
        [1] + x + [1] for x in tokenized_inputs["attention_mask"]
    ]
    labels = copy.deepcopy(tokenized_inputs["input_ids"])
    for instr, lbl in zip(tokenized_instructions["input_ids"], labels):
        lbl[: len(instr)] = [-100] * len(instr)

    tokenized_inputs["input_ids"] = pad(
        tokenized_inputs["input_ids"], tokenizer.pad_token_id, max_length, padding_side
    )
    tokenized_inputs["attention_mask"] = pad(
        tokenized_inputs["attention_mask"], 0, max_length, padding_side
    )
    labels = pad(labels, -100, max_length, padding_side)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,
    }


@DatasetCache(hash_keys=["task"])
def load_scone(config: DictConfig):
    splits = ["train", "dev", "test"]
    tasks = ["alchemy", "scene", "tangrams"]

    def tsv_to_dict_of_lists(file_path):
        with open(file_path, "r", newline="") as tsv_file:
            reader = csv.reader(tsv_file, delimiter="\t")

            # Read the first row to determine the number of columns
            first_row = next(reader)
            num_columns = len(first_row)

            # Generate headers
            headers = ["ID", "WORLD_0"]
            for i in range(1, (num_columns - 2) // 2 + 1):
                headers.extend([f"UTTERANCE_{i}", f"WORLD_{i}"])

            # Create a dictionary to store the lists
            result_dict = {header: [] for header in headers}

            # Reset the file pointer to the beginning
            tsv_file.seek(0)

            # Process each row
            for row in reader:
                for i, value in enumerate(row):
                    if i < len(headers):
                        result_dict[headers[i]].append(value)

        return result_dict

    task_datasets = defaultdict(list)

    for split in splits:
        for task in tasks:
            ds = datasets.Dataset.from_dict(
                tsv_to_dict_of_lists(f"{config.task.name_or_path}/{task}-{split}.tsv")
            )
            ds = ds.add_column("task", [task] * len(ds))
            task_datasets[split].append(ds)

    for split, ds_list in task_datasets.items():
        task_datasets[split] = datasets.concatenate_datasets(ds_list)

    scone_dataset = datasets.DatasetDict(task_datasets)

    def extract_index_and_number(input_string):
        pattern = r"^(\d+):(.*)$"
        match = re.search(pattern, input_string)

        if match:
            index = match.group(1)
            number = match.group(2)
            return index, number
        else:
            return None

    def alchemy_state_to_nl(state: str):
        beakers = list(map(lambda x: extract_index_and_number(x), state.split(" ")))

        def color_sequence_to_instruction(sequence):
            # Count the occurrences of each color
            color_counts = Counter(sequence.lower())
            # Create a list of color instructions
            instructions = []
            for color, count in color_counts.items():
                full_color_name = COLOR_MAP[color]
                instructions.append(f"{count} {full_color_name}")

            # Join the instructions
            if len(instructions) == 1:
                return instructions[0]
            else:
                return "{" + ", ".join(instructions) + "}"

        def to_nl(x):
            i, s = x
            if s[1] == "_":
                return f"the {NUM2WORD[i + 1]} beaker is empty"
            return f"the {NUM2WORD[i + 1]} beaker has {color_sequence_to_instruction(s[1])}"

        return ", ".join(map(to_nl, enumerate(beakers)))

    def scene_state_to_nl(state: str):
        positions = list(map(lambda x: extract_index_and_number(x), state.split(" ")))

        def to_nl(x):
            i, s = x
            if s[1][0] == "_":
                return f"the {NUM2WORD[i + 1]} position is empty"
            hat = COLOR_MAP[s[1][1]] if s[1][1] != "_" else "no"
            return f"the {NUM2WORD[i + 1]} position is occupied by a person with a {COLOR_MAP[s[1][0]]} shirt and {hat} hat"

        return ", ".join(map(to_nl, enumerate(positions)))

    def tangram_state_to_nl(state: str):
        if all(not x for x in state.split(" ")):
            return "no tangrams present"

        tangrams = list(map(lambda x: extract_index_and_number(x), state.split(" ")))

        def to_nl(x):
            i, s = x
            if s[1] == "_":
                return f"the {NUM2WORD[i + 1]} tangram is not placed"
            return f"{NUM2WORD[i + 1]} object id={s[1]}"

        return ", ".join(map(to_nl, enumerate(tangrams)))

    def sequence_to_instruction(
        example: dict,
        min_turn_limit: int,
        max_turn_limit: int,
        samples_per_sequence: int,
    ):
        # batch size 1
        example = {k: v[0] for k, v in example.items()}

        if example["task"] == "alchemy":
            nl_fn = alchemy_state_to_nl
        elif example["task"] == "tangrams":
            nl_fn = tangram_state_to_nl
        elif example["task"] == "scene":
            nl_fn = scene_state_to_nl

        limit = len([k for k in example.keys() if k.startswith("WORLD_")])

        world_states = [nl_fn(example[f"WORLD_{i}"]) for i in range(0, limit)]
        utterances = [example[f"UTTERANCE_{i}"] for i in range(1, limit)]
        utterances.insert(0, "")
        utterances.append("")

        samples_per_sequence = min(
            samples_per_sequence, max_turn_limit - min_turn_limit + 1
        )

        instructions, outputs = [], []

        if config.task.mode == "editor":
            # Instruction = (state,utterance) pairs and output is next state
            turn_limits = random.sample(
                range(min_turn_limit, max_turn_limit + 1), k=samples_per_sequence
            )
            target_inputs = []

            for turn_limit in turn_limits:
                instruction = []
                target_input = None
                output = None
                for i, state in enumerate(world_states):
                    utterance = utterances[i + 1]
                    if i + 1 < min(limit, turn_limit):
                        instruction.extend([state, utterance])
                    else:
                        target_input = utterances[i]
                        instruction.pop()
                        output = state
                        break

                instructions.append("\n".join(instruction))
                outputs.append(output)
                target_inputs.append(target_input)

            return {
                "editor_context": instructions,
                "target_input": target_inputs,
                "target": outputs,
                "task": [example["task"]] * len(instructions),
            }

        elif config.task.mode == "sft":
            # Instruction = single utterance and output is next state
            instructions = utterances[1:-1]
            outputs = world_states[1:]

            return {
                "instruction": instructions,
                "target": outputs,
                "task": [example["task"]] * len(instructions),
            }

    scone_processed = scone_dataset.map(
        partial(
            sequence_to_instruction,
            min_turn_limit=config.task.min_turn_limit,
            max_turn_limit=config.task.max_turn_limit,
            samples_per_sequence=config.task.samples_per_sequence,
        ),
        num_proc=os.cpu_count() if not config.debug else 1,
        batched=True,
        batch_size=1,
        load_from_cache_file=False,
        remove_columns=[
            c
            for c in scone_dataset["train"].column_names
            if c not in ["instruction", "target", "task"]
        ],
    )

    scone_filtered = scone_processed.filter(lambda x: x["task"] in config.task.domains)

    # tokenized
    if config.task.mode == "sft":
        tokenized_scone = scone_filtered.map(
            partial(
                tokenize_sft,
                tokenizer=get_tokenizer(config.model.name_or_path),
                max_length=config.model.max_length,
                instruction_col="instruction",
                target_col="target",
                padding_side=config.data.padding_side,
            ),
            batched=True,
            load_from_cache_file=False,
        )
    else:
        tokenized_scone = scone_filtered.map(
            partial(
                tokenize_editor_eval,
                tokenizer=get_tokenizer(config.model.name_or_path),
                max_length=config.model.max_length,
                editor_token_limit=config.task.editor_token_limit,
                editor_input_col="editor_context",
                target_input_col="target_input",
                target_col="target",
            ),
            batched=True,
            batch_size=1,
            load_from_cache_file=False,
        )

    # remap splits
    tokenized_scone = datasets.DatasetDict(
        {
            "train": tokenized_scone["train"],
            "test": tokenized_scone["test"],
            "val": tokenized_scone["dev"],
        }
    )

    return tokenized_scone


@DatasetCache(hash_keys=["task"])
def load_counterfact(config: DictConfig):
    dataset = datasets.load_from_disk(config.task.name_or_path)
    tokenizer = get_tokenizer(
        config.model.name_or_path, padding_side=config.data.padding_side
    )

    def preprocess(batch):
        examples = defaultdict(list)
        for requested_rewrite, continuations in zip(
            batch["requested_rewrite"], batch["generation_continuations"]
        ):
            instruction = (
                requested_rewrite["prompt"].format(requested_rewrite["subject"])
                + " "
                + requested_rewrite["target_new"]["str"]
                + ". "
            )
            examples["editor"].extend([instruction] * len(continuations))
            examples["target"].extend(continuations)

        return examples

    processed_data = dataset.map(
        preprocess,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=[
            c for c in dataset.column_names if c not in ["editor", "target"]
        ],
        load_from_cache_file=False,
    )
    tokenized_data = processed_data.map(
        partial(
            tokenize_editor,
            tokenizer=tokenizer,
            max_length=config.model.max_length,
            editor_token_limit=config.task.editor_token_limit,
            instruction_col="editor",
            target_col="target",
        ),
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
        remove_columns=processed_data.column_names,
    )
    return tokenized_data


def shuffle_and_select(
    dataset,
    split: Literal["val", "test", "train"],
    test_split: float,
    val_split: float,
    seed: int,
    do_eval: bool = False,
    is_split: bool = False,
):
    """Shuffle and select a split of a flat dataset."""
    # shuffle and take split according to seed
    dataset = dataset.shuffle(seed=seed)

    # get number of examples parsed from split
    match = re.search(r"(\w+)\[\:(\d+)(\%)*\]", split)

    if match is None:
        split, split_num_examples = split, None
    else:
        split, split_num_examples = match.group(1), match.group(2)

    # create train/test/val splits
    if not is_split:
        _split_dataset = dataset.train_test_split(test_size=test_split + val_split)
    else:
        _split_dataset = dataset

    if "train" in split:
        dataset = _split_dataset["train"]
    else:
        if not is_split:
            test_val_split = _split_dataset["test"].train_test_split(
                test_size=val_split / (test_split + val_split)
            )
            if "val" in split:
                dataset = test_val_split["train"]
            else:
                dataset = test_val_split["test"]
        else:
            dataset = _split_dataset[split]

    if split_num_examples:
        if "%" in split:
            split_num_examples = math.floor(
                int(split_num_examples[:-1]) * len(dataset) / 100
            )
        else:
            split_num_examples = int(split_num_examples)

        # select first split_num_examples from dataset
        return dataset.select(range(split_num_examples))

    return dataset


def get_dataloader(
    dataset: datasets.Dataset, config: DictConfig, split: str
) -> DataLoader:
    # Mike: I was getting a device error from the RNG generator being on CPU by default before before so I added this and imported torch
    # generator = torch.Generator(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return DataLoader(
        dataset,
        batch_size=config.train.train_batch_size
        if "train" in split
        else config.train.validation_batch_size,
        shuffle=True if "train" in split else False,
        collate_fn=partial(transformers.default_data_collator, return_tensors="pt"),
        # generator=generator, #also added this line, see comment above
    )


def get_task(config: DictConfig, suffix: str, split: str) -> Callable:
    dataset_load_fn = globals().get("load_" + suffix)
    dataset = dataset_load_fn(config)

    if config.data.n_examples > 0:
        split = f"{split}[:{config.data.n_examples}]"

    return shuffle_and_select(
        dataset,
        split,
        test_split=config.data.test_split,
        val_split=config.data.val_split,
        seed=config.seed,
        is_split=config.task.is_split,
    )
