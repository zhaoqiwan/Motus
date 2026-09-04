#!/usr/bin/env python3
"""
Generate task-level WAN T5 embedding caches for a LeRobot v3 dataset.

Read task instructions from meta/tasks.parquet, encode each unique
instruction with WAN's pretrained UMT5 encoder, and save:

    t5_embedding/task_XXXXXX.pt

Existing cache files are skipped unless --overwrite is specified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pyarrow.parquet as pq
import torch


def _load_task_prompts(dataset_root: Path) -> List[Tuple[int, str]]:
    """Read task_index and instruction pairs from a LeRobot v3 dataset."""
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.parquet not found: {tasks_path}")

    table = pq.read_table(tasks_path)

    prompt_column = (
        "task"
        if "task" in table.column_names
        else "__index_level_0__"
    )

    task_indices = table["task_index"].to_pylist()
    prompts = table[prompt_column].to_pylist()

    return [
        (int(task_index), str(prompt))
        for task_index, prompt in zip(task_indices, prompts)
    ]


def _init_wan_t5_encoder(
    wan_path: str,
    device: str,
    text_len: int = 512,
) -> Any:
    """
    Initialize WAN T5EncoderModel.
    We mirror the initialization used in Motus inference scripts.
    """
    try:
        from Motus.bak.wan.modules.t5 import T5EncoderModel  # type: ignore
    except Exception:
        # Fallback: add bak path similarly to inference scripts
        bak_root = str((Path(__file__).resolve().parents[2] / "bak").resolve())
        if bak_root not in sys.path:
            sys.path.insert(0, bak_root)
        from wan.modules.t5 import T5EncoderModel  # type: ignore

    ckpt = os.path.join(wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"T5 checkpoint not found: {ckpt}")
    if not os.path.exists(tok):
        raise FileNotFoundError(f"T5 tokenizer dir not found: {tok}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=int(text_len),
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )


def _encode_t5(encoder: Any, instruction: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        out = encoder([instruction], device)
    if isinstance(out, list):
        emb = out[0]
    elif isinstance(out, torch.Tensor):
        emb = out
    else:
        raise ValueError(f"Unexpected T5 encoder output type: {type(out)}")

    # Normalize to [S, D] tensor on CPU
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.detach().cpu()



def main():
    parser = argparse.ArgumentParser(
        description="Generate task-level WAN T5 caches for a LeRobot v3 dataset"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local LeRobot dataset root containing meta/tasks.parquet.",
    )
    parser.add_argument(
        "--wan_path",
        type=str,
        default=None,
        help="Base path that contains Wan2.2-TI2V-5B/ (can also use env WAN_PATH/WAN_ROOT).",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--text_len", type=int, default=512, help="T5 text_len (default: 512)")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding", help="Cache folder name under dataset root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task_XXXXXX.pt files",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wan_path = args.wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
    if not wan_path:
        raise ValueError("WAN path not provided. Use --wan_path or set WAN_PATH/WAN_ROOT.")

    dataset_root = args.root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    task_prompts = _load_task_prompts(dataset_root)

    out_dir = dataset_root / args.t5_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = None
    created = 0
    skipped = 0

    for task_index, instruction in task_prompts:
        output_path = out_dir / f"task_{task_index:06d}.pt"

        if output_path.exists() and not args.overwrite:
            print(f"Skip existing cache: {output_path.name}")
            skipped += 1
            continue

        if not instruction.strip():
            raise ValueError(f"Empty instruction for task_index={task_index}")

        if encoder is None:
            print(f"Loading WAN T5 encoder from {wan_path} on {device} ...")
            encoder = _init_wan_t5_encoder(
                wan_path=wan_path,
                device=device,
                text_len=args.text_len,
            )

        print(f"Encoding task {task_index}: {instruction}")
        embedding = _encode_t5(
            encoder,
            instruction,
            device=device,
        )

        torch.save(embedding, output_path)
        print(f"Saved: {output_path} shape={tuple(embedding.shape)}")
        created += 1

    print(f"Done. created={created}, skipped={skipped}")
    print(f"Dataset root: {dataset_root}")

if __name__ == "__main__":
    main()
