import hashlib
import inspect
import json
import math
import os
import re
from typing import Any, Callable, List, Literal

import datasets
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from helpers import ROOT_DIR, get_tokenizer
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
def load_nouns(config: DictConfig):
    raise NotImplementedError


@DatasetCache(hash_keys=["task"])
def load_wikipedia(config: DictConfig):
    assert config.task.name == "wikipedia", "task must be 'wikipedia'"

    dataset = datasets.load_dataset("abokbot/wikipedia-first-paragraph", split="train")
    tokenizer = get_tokenizer(config.model.name_or_path)

    assert tokenizer.padding_side == "right", "padding_side must be 'right'"

    def extract_sentences(texts: dict):
        first_sentences = []
        second_sentences = []
        third_sentences = []

        for text in texts["text"]:
            # Split the text into sentences
            sentences = text.split(". ")
            # Extract the first sentence
            first_sentence = sentences[0] if len(sentences) > 0 else ""
            first_sentences.append(first_sentence + "." if first_sentence else "")
            # Extract the second sentence, if it exists
            second_sentence = sentences[1] if len(sentences) > 1 else ""
            second_sentences.append(second_sentence + "." if second_sentence else "")
            # Extract the third sentence, if it exists
            third_sentence = sentences[2] if len(sentences) > 2 else ""
            third_sentences.append(third_sentence + "." if third_sentence else "")

        return {
            "first_sentence": first_sentences,
            "second_sentence": second_sentences,
            "third_sentence": third_sentences,
        }

    new_dataset = dataset.map(extract_sentences, batched=True, num_proc=os.cpu_count())

    new_dataset = new_dataset.remove_columns(["text", "url", "id"])
    new_dataset = new_dataset.filter(
        lambda row: len(row["first_sentence"]) >= 5
        and len(row["second_sentence"]) >= 10
        and (0 < len(row["third_sentence"]) <= 100),
        num_proc=os.cpu_count(),
    )

    def tokenize(row_batch: dict, tokenizer=None):
        # Compose second_sentences and third_sentences
        followup_text = [
            second_sentence + " " + third_sentence
            for second_sentence, third_sentence in zip(
                row_batch["second_sentence"], row_batch["third_sentence"]
            )
        ]
        # Select the first 500 characters
        followup_text = [
            text[: config.task.followup_char_limit] for text in followup_text
        ]

        target_inputs = tokenizer(
            followup_text,
            add_special_tokens=False,
            max_length=config.task.editor_token_limit,
            padding="max_length",
            truncation=True,
        )
        editor_inputs = tokenizer(
            row_batch["first_sentence"],
            max_length=config.model.max_length,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
        )

        return {
            **{"editor_" + k: v for k, v in editor_inputs.items()},
            **{"target_" + k: v for k, v in target_inputs.items()},
        }

    new_dataset = new_dataset.map(
        tokenize,
        batched=True,
        num_proc=os.cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
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


def shuffle_and_select(
    dataset,
    split: Literal["val", "test", "train"],
    test_split: float,
    val_split: float,
    seed: int,
    do_eval: bool = False,
):
    """Shuffle and select a split of a flat dataset."""
    # shuffle and take split according to seed
    dataset = dataset.shuffle(seed=seed)

    if split == "train" and not do_eval:
        return dataset

    # get number of examples parsed from split
    match = re.search(r"(\d+)(\%)*", split)

    if match is None:
        split_num_examples = None
    else:
        split_num_examples = match.group(0)

    # create train/test/val splits
    _split_dataset = dataset.train_test_split(test_size=test_split + val_split)

    if "train" in split:
        dataset = _split_dataset["train"]
    else:
        test_val_split = _split_dataset["test"].train_test_split(
            test_size=val_split / (test_split + val_split)
        )
        if "val" in split:
            dataset = test_val_split["train"]
        else:
            dataset = test_val_split["test"]

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
    )
