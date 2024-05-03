import datasets
import pandas as pd
from omegaconf import DictConfig
from transformers import AutoTokenizer

task_registry = {}


def register_task():
    """Registers an editing task"""

    def wrapper(func):
        task_registry[func.__name__] = func
        return func

    return wrapper


@register_task("nouns")
def load_nouns(config: DictConfig):
    pass


@register_task("wikipedia")
def load_wikipedia(config: DictConfig):
    raw_dataset = datasets.load_dataset("abokbot/wikipedia-first-paragraph")

    def extract_sentences(texts):
        first_sentences = []
        second_sentences = []
        third_sentences = []

        for text in texts:
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

        sentences_dict = {
            "first_sentences": first_sentences,
            "second_sentences": second_sentences,
            "third_sentences": third_sentences,
        }
        return sentences_dict

    def splitsentences(datarow):
        return extract_sentences(datarow["text"])

    new_dataset = raw_dataset["train"].map(splitsentences, batched=True)
    wikipedia_dataset = new_dataset.remove_columns(["text", "url", "id"])

    def is_not_empty(row):
        # Check if either first_sentence or second_sentence is empty
        return (
            row["first_sentences"] != ""
            and row["second_sentences"] != ""
            and row["third_sentences"] != ""
        )

    filtered_dataset = wikipedia_dataset.filter(is_not_empty)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_followup(row):
        # Compose second_sentences and third_sentences
        followup_text = row["second_sentences"] + " " + row["third_sentences"]

        # Select the first 500 characters
        followup_text = followup_text[:500]

        # Tokenize the followup text
        tokenized_followup = tokenizer.encode(followup_text)

        # Check if the resulting tokenized list is still less than 50 tokens
        if len(tokenized_followup) < 50:
            # Pad with token 50256 to reach a length of 50 tokens
            tokenized_followup = tokenized_followup + [50256] * (
                50 - len(tokenized_followup)
            )

        return tokenized_followup


def edit_data_collate_fn():
    pass
