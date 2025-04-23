from __future__ import annotations
import json
from transformers import LlamaTokenizer, LlamaTokenizerFast, AutoTokenizer
from typing import Optional
import pydantic
"""
Huggingface tokenizer is a little sus, so we have this specifically.
"""

class LlamaTokens(pydantic.BaseModel):
    #### IDs ####
    bos_id: int
    eos_id: int  # end of text
    finetune_right_pad_id: int
    start_header_id: int
    end_header_id: int
    eom_id: int
    eot_id: int
    python_tag_id: int
    #### Strings ####
    bos: str
    eos: str
    finetune_right_pad: str
    start_header: str
    end_header: str
    eom: str
    eot: str
    python_tag: str


def get_llama_tokens(
    tokenizer: Optional[LlamaTokenizer] = None,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
) -> LlamaTokens:
    # TODO(Adriano) we should enforce llama tokenizer...
    tokenizer = (
        tokenizer
        if tokenizer is not None
        else AutoTokenizer.from_pretrained(model_name)
    )
    # assert isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)), f"Tokenizer must be a LlamaTokenizer or LlamaTokenizerFast, got {type(tokenizer)}"; fmt: skip
    special_tokens = """
    <|begin_of_text|>
    <|end_of_text|>
    <|finetune_right_pad_id|>
    <|start_header_id|>
    <|end_header_id|>
    <|eom_id|>
    <|eot_id|>
    <|python_tag|>
    """.split(
        "\n"
    )
    special_tokens = [x.strip() for x in special_tokens if x.strip()]
    assert all(isinstance(x, str) for x in special_tokens)
    special_token_ids = [
        tokenizer(x, add_special_tokens=False)["input_ids"] for x in special_tokens
    ]
    assert all(isinstance(x, list) for x in special_token_ids)
    assert all(isinstance(x[0], int) for x in special_token_ids)
    assert all(len(x) == 1 for x in special_token_ids)
    name2id: dict[str, int] = {
        name: ids[0] for name, ids in zip(special_tokens, special_token_ids)
    }
    return LlamaTokens(
        #### IDs ####
        bos_id=name2id["<|begin_of_text|>"],
        eos_id=name2id["<|end_of_text|>"],
        finetune_right_pad_id=name2id["<|finetune_right_pad_id|>"],
        start_header_id=name2id["<|start_header_id|>"],
        end_header_id=name2id["<|end_header_id|>"],
        eom_id=name2id["<|eom_id|>"],
        eot_id=name2id["<|eot_id|>"],
        python_tag_id=name2id["<|python_tag|>"],
        #### Strings ####
        bos="<|begin_of_text|>",
        eos="<|end_of_text|>",
        finetune_right_pad="<|finetune_right_pad_id|>",
        start_header="<|start_header_id|>",
        end_header="<|end_header_id|>",
        eom="<|eom_id|>",
        eot="<|eot_id|>",
        python_tag="<|python_tag|>",
    )


if __name__ == "__main__":
    tokens = get_llama_tokens()
    print(tokens.model_dump_json(indent=4))  # Debug
