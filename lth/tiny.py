"""A randomly initialised 2-layer Llama and a character tokenizer.

Used by the tests and by `--model tiny-random` to smoke-test the whole pipeline
without downloading anything. Accuracy numbers from it are meaningless.
"""
from __future__ import annotations

import torch


class CharTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.chars = [chr(i) for i in range(32, 127)] + ["\n"]
        self.stoi = {c: i + 4 for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 4

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.stoi.get(c, self.unk_token_id) for c in text]

    def __call__(self, text: str, add_special_tokens: bool = False, **_) -> dict:
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i >= 4:
                out.append(self.chars[i - 4])
            elif not skip_special_tokens:
                out.append(f"<{i}>")
        return "".join(out)


def build_tiny(seed: int = 0, device: str = "cpu"):
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = CharTokenizer()
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    return LlamaForCausalLM(config).to(device), tok
