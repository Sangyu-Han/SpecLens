from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, DistributedSampler
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Minimal text dataset that exposes sample_id for provenance tracking."""

    def __init__(self, samples: Sequence[str]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "text": self.samples[index],
            "sample_id": int(index),
        }


def _load_texts_from_file(path: Path, *, text_key: str = "text") -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        texts: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if text_key not in record:
                    raise KeyError(f"Missing '{text_key}' in record from {path}")
                texts.append(str(record[text_key]))
        return texts

    # Plain text: one prompt per line
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _resolve_samples(dataset_cfg: Dict[str, Any]) -> List[str]:
    text_key = dataset_cfg.get("text_key", "text")
    samples: List[str] = []

    inline = dataset_cfg.get("texts") or dataset_cfg.get("data")
    if inline:
        if isinstance(inline, (list, tuple)):
            for item in inline:
                if isinstance(item, str):
                    samples.append(item)
                elif isinstance(item, dict) and text_key in item:
                    samples.append(str(item[text_key]))
        elif isinstance(inline, str):
            samples.append(inline)

    path_field = dataset_cfg.get("path") or dataset_cfg.get("file") or dataset_cfg.get("jsonl")
    if path_field:
        texts = _load_texts_from_file(Path(path_field).expanduser(), text_key=text_key)
        samples.extend(texts)

    if not samples:
        raise ValueError("No text samples found. Provide 'texts' or a valid 'path'/'jsonl'.")

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    return samples


def make_text_collate_fn(
    tokenizer,
    *,
    max_length: Optional[int] = None,
    padding: str | bool = "longest",
    truncation: bool = True,
    pad_to_multiple_of: Optional[int] = None,
    add_special_tokens: bool = True,
    prepend_bos: bool = False,
) -> Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]:
    """
    Build a collate_fn that tokenizes text batches and preserves sample_id.
    """

    bos_id = tokenizer.bos_token_id

    def collate(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        items = list(batch)
        texts = [str(item["text"]) for item in items]
        sample_ids = torch.tensor([int(item.get("sample_id", idx)) for idx, item in enumerate(items)], dtype=torch.long)

        tokenized = tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens and not prepend_bos,
            return_tensors="pt",
            pad_to_multiple_of=pad_to_multiple_of,
        )

        if prepend_bos and bos_id is not None:
            input_ids = tokenized["input_ids"]
            attn = tokenized.get("attention_mask", torch.ones_like(input_ids))
            bos_col = torch.full((input_ids.shape[0], 1), bos_id, dtype=input_ids.dtype)
            input_ids = torch.cat([bos_col, input_ids], dim=1)
            attn = torch.cat([torch.ones_like(bos_col), attn], dim=1)
            tokenized["input_ids"] = input_ids
            tokenized["attention_mask"] = attn

        tokenized["sample_id"] = sample_ids
        tokenized["text"] = texts
        return tokenized

    return collate


def build_text_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
    tokenizer_name: str,
    device: torch.device | None = None,
    **_,
) -> Dict[str, Any]:
    """
    Construct a text dataset + sampler + collate_fn for Gemma models.
    """
    samples = _resolve_samples(dataset_cfg)
    dataset = TextDataset(samples)

    # Tokenizer setup
    tok_kwargs = {}
    if dataset_cfg.get("trust_remote_code") is not None:
        tok_kwargs["trust_remote_code"] = bool(dataset_cfg.get("trust_remote_code"))
    if dataset_cfg.get("revision") is not None:
        tok_kwargs["revision"] = dataset_cfg.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tok_kwargs)
    tokenizer.padding_side = dataset_cfg.get("padding_side", tokenizer.padding_side)

    # Collate with tokenizer
    collate_fn = make_text_collate_fn(
        tokenizer,
        max_length=dataset_cfg.get("max_length"),
        padding=dataset_cfg.get("padding", "longest"),
        truncation=bool(dataset_cfg.get("truncation", True)),
        pad_to_multiple_of=dataset_cfg.get("pad_to_multiple_of"),
        add_special_tokens=bool(dataset_cfg.get("add_special_tokens", True)),
        prepend_bos=bool(dataset_cfg.get("prepend_bos", False)),
    )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(dataset_cfg.get("shuffle", True)),
            drop_last=bool(dataset_cfg.get("drop_last", False)),
        )

    return {
        "dataset": dataset,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "tokenizer": tokenizer,
    }


__all__ = [
    "TextDataset",
    "make_text_collate_fn",
    "build_text_dataset",
]
