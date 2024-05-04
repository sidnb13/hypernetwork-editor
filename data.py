import hashlib
import json
import math
import os
import re
from typing import Any, Callable, List, Literal

import datasets
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from helpers import get_tokenizer
from logger import get_logger

logger = get_logger(__name__)


class DatasetCache:
    def __init__(
        self,
        exclude_argnames: List[str],
        cache_dir: str | os.PathLike = "assets/data/cache",
    ) -> None:
        self.exclude_argnames = exclude_argnames
        self.cache_dir = cache_dir

    def __call__(self, func) -> Any:
        def wrapper(*args, **kwargs):
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            def get_pruned_dict(obj):
                prune = {}
                for k, v in obj.items():
                    if any(x in k for x in self.exclude_argnames):
                        continue
                    if isinstance(v, (dict, DictConfig)):
                        prune[k] = get_pruned_dict(dict(v))
                    else:
                        prune[k] = v
                return prune

            signature = hashlib.md5(
                json.dumps(get_pruned_dict(dict(args[0]))).encode("utf-8")
            ).hexdigest()
            is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
            rank = 0 if is_rank0 else dist.get_rank()

            def barrier():
                return dist.barrier() if dist.is_initialized() else None

            cache_folder = os.path.join(self.cache_dir, f"{func.__name__}_{signature}")
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

        return wrapper


@DatasetCache(
    exclude_argnames=["n_examples", "test_split", "val_split", "seed", "batch_size"]
)
def load_nouns(config: DictConfig):
    pass


@DatasetCache(
    exclude_argnames=["n_examples", "test_split", "val_split", "seed", "batch_size"]
)
def load_wikipedia(config: DictConfig):
    assert config.task.name == "wikipedia", "task must be 'wikipedia'"

    dataset = datasets.load_dataset("abokbot/wikipedia-first-paragraph", split="train")
    tokenizer = get_tokenizer(config.model.name_or_path)

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

        # Tokenize the followup text
        editor_inputs = tokenizer(
            followup_text,
            add_special_tokens=False,
            max_length=config.task.editor_token_limit,
            padding="max_length",
            truncation=True,
        )
        # Tokenize the target text
        target_inputs = tokenizer(
            row_batch["first_sentence"],
            max_length=config.model.max_length,
            add_special_tokens=False,
            padding="max_length",
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
):
    """Shuffle and select a split of a flat dataset."""
    # shuffle and take split according to seed
    dataset = dataset.shuffle(seed=seed)
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
    return DataLoader(
        dataset,
        batch_size=config.data.train_batch_size
        if "train" in split
        else config.data.eval_batch_size,
        shuffle=True if "train" in split else False,
    )


def get_task(config: DictConfig, suffix: str, split: str) -> Callable:
    dataset = globals().get("load_" + suffix)(config)

    if config.data.n_examples > 0:
        split = f"{split}[:{config.data.n_examples}]"

    return shuffle_and_select(
        dataset,
        split,
        test_split=config.data.test_split,
        val_split=config.data.val_split,
        seed=config.seed,
    )
