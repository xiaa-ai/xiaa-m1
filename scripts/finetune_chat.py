"""Supervised fine-tuning script for Xiaa M1 chat behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PretrainedConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xiaa.chat_format import get_assistant_mask
from xiaa.config import XiaaConfig
from xiaa.model import XiaaM1
from xiaa.tokenizer import XiaaTokenizer


class XiaaHFConfig(PretrainedConfig):
    """Minimal Hugging Face config wrapper for Xiaa M1."""

    model_type = "xiaa-m1"

    def __init__(self, xiaa_config: dict[str, Any], **kwargs: Any) -> None:
        """Store the serialized Xiaa config inside the HF config."""
        super().__init__(**kwargs)
        self.xiaa_config = xiaa_config


class XiaaHFModel(PreTrainedModel):
    """Hugging Face wrapper around the native XiaaM1 model."""

    config_class = XiaaHFConfig

    def __init__(self, config: XiaaHFConfig) -> None:
        """Initialize the wrapped XiaaM1 model from HF config."""
        super().__init__(config)
        model_cfg = XiaaConfig(**config.xiaa_config)
        self.model = XiaaM1(model_cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Run a forward pass through XiaaM1 and return HF-style outputs."""
        logits, loss = self.model(input_ids, labels)
        return {"logits": logits, "loss": loss}


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized SFT samples with assistant-only loss masking."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: XiaaTokenizer,
        seq_len: int,
    ) -> None:
        """Load and tokenize SFT data from JSONL."""
        self.examples: list[dict[str, torch.Tensor]] = []
        self.seq_len = seq_len

        if not jsonl_path.exists():
            raise FileNotFoundError(f"SFT data file not found: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue

                input_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(input_ids) < 2:
                    continue

                mask = get_assistant_mask(input_ids, tokenizer)
                x_ids = input_ids[:-1]
                y_ids = input_ids[1:]
                y_mask = mask[1:]

                y_labels = [token if m == 1 else -1 for token, m in zip(y_ids, y_mask)]

                if len(x_ids) > seq_len:
                    x_ids = x_ids[:seq_len]
                    y_labels = y_labels[:seq_len]
                else:
                    pad_len = seq_len - len(x_ids)
                    x_ids.extend([0] * pad_len)
                    y_labels.extend([-1] * pad_len)

                if not any(label != -1 for label in y_labels):
                    continue

                self.examples.append(
                    {
                        "input_ids": torch.tensor(x_ids, dtype=torch.long),
                        "labels": torch.tensor(y_labels, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        """Return number of SFT examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one tokenized training example."""
        return self.examples[index]


def get_device() -> torch.device:
    """Select training device with mps > cuda > cpu priority."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_best_pretrain_checkpoint(checkpoint_dir: Path) -> Path:
    """Return best available pretraining checkpoint path."""
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    if candidates:
        return candidates[-1]

    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest

    raise FileNotFoundError(
        "No pretraining checkpoint found. Expected checkpoints/latest.pt or step_*.pt"
    )


def save_hf_export(model: XiaaM1, cfg: XiaaConfig, output_dir: Path) -> None:
    """Save model in Hugging Face format using a PreTrainedModel wrapper."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_config = XiaaHFConfig(xiaa_config=cfg.__dict__.copy())
    hf_model = XiaaHFModel(hf_config)
    hf_model.model.load_state_dict(model.state_dict())
    hf_model.save_pretrained(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for SFT."""
    parser = argparse.ArgumentParser(description="SFT fine-tuning for Xiaa M1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    return parser


def main() -> None:
    """Run supervised fine-tuning on chat-formatted JSONL data."""
    args = build_arg_parser().parse_args()

    checkpoint_dir = REPO_ROOT / "checkpoints"
    chat_data_path = REPO_ROOT / "data" / "chat" / "sft_data.jsonl"
    final_sft_path = checkpoint_dir / "xiaa_m1_sft.pt"
    hf_export_dir = checkpoint_dir / "xiaa_m1_hf"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chat_data_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = XiaaConfig()
    cfg.max_seq_len = args.seq_len

    tokenizer = XiaaTokenizer(REPO_ROOT / "xiaa" / "xiaa_tokenizer.model")
    dataset = SFTDataset(chat_data_path, tokenizer=tokenizer, seq_len=args.seq_len)
    if len(dataset) == 0:
        raise RuntimeError("No valid SFT examples found after tokenization/masking.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = get_device()
    print(f"Using device: {device}")

    model = XiaaM1(cfg).to(device)

    pretrain_checkpoint = find_best_pretrain_checkpoint(checkpoint_dir)
    print(f"Loading pretraining checkpoint: {pretrain_checkpoint}")
    state = torch.load(pretrain_checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels)
            if loss is None:
                raise RuntimeError("Model returned None loss for SFT batch.")

            loss.backward()
            optimizer.step()

            global_step += 1
            print(
                f"SFT Epoch {epoch + 1}/{args.epochs} | "
                f"Step {global_step} | Loss: {float(loss.item()):.4f}"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__.copy(),
            "epochs": args.epochs,
            "global_step": global_step,
        },
        final_sft_path,
    )
    print(f"Saved SFT checkpoint: {final_sft_path}")

    save_hf_export(model=model, cfg=cfg, output_dir=hf_export_dir)
    print(f"Saved Hugging Face export: {hf_export_dir}")


if __name__ == "__main__":
    main()
