from transformers import AutoTokenizer


def get_tokenizer(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token_id = tok.eos_token_id
    return tok
