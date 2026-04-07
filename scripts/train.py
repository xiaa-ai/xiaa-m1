"""Pre-train Xiaa M1 from binary uint16 shards using raw PyTorch."""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from xiaa.config import XiaaConfig
from xiaa.model import XiaaM1


def get_device() -> torch.device:
	"""Select training device with mps > cuda > cpu priority."""
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def cosine_lr(
	step: int,
	max_steps: int,
	warmup_steps: int = 2000,
	max_lr: float = 3e-4,
	min_lr: float = 3e-5,
) -> float:
	"""Compute warmup + cosine learning rate for a training step."""
	if step < warmup_steps:
		return max_lr * float(step + 1) / float(max(1, warmup_steps))

	progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
	progress = min(max(progress, 0.0), 1.0)
	cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
	return min_lr + (max_lr - min_lr) * cosine_decay


class ShardBatchSampler:
	"""Randomly sample (x, y) training windows from uint16 shard files."""

	def __init__(self, shard_dir: Path, seq_len: int) -> None:
		"""Load shard metadata and initialize memmap cache."""
		self.seq_len = seq_len
		self.shard_paths = sorted(shard_dir.glob("shard_*.bin"))
		if not self.shard_paths:
			raise FileNotFoundError(
				f"No shard files found under {shard_dir}. Run scripts/prepare_data.py first."
			)
		self._memmaps: dict[Path, np.memmap] = {}

	def get_memmap(self, shard_path: Path) -> np.memmap:
		"""Return a cached memmap view for a shard path."""
		if shard_path not in self._memmaps:
			self._memmaps[shard_path] = np.memmap(shard_path, dtype=np.uint16, mode="r")
		return self._memmaps[shard_path]

	def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""Sample one batch of contiguous token windows from random shards."""
		x_batch = torch.empty((batch_size, self.seq_len), dtype=torch.long)
		y_batch = torch.empty((batch_size, self.seq_len), dtype=torch.long)

		for index in range(batch_size):
			shard_path = random.choice(self.shard_paths)
			shard = self.get_memmap(shard_path)
			if shard.shape[0] <= self.seq_len:
				raise ValueError(
					f"Shard too small for seq_len={self.seq_len}: {shard_path}"
				)

			start = random.randint(0, int(shard.shape[0] - self.seq_len - 1))
			window = np.asarray(shard[start : start + self.seq_len + 1], dtype=np.int64)
			x_batch[index] = torch.from_numpy(window[:-1].copy())
			y_batch[index] = torch.from_numpy(window[1:].copy())

		return x_batch, y_batch


def load_checkpoint_if_available(
	model: XiaaM1,
	optimizer: torch.optim.Optimizer,
	checkpoint_path: Path,
	device: torch.device,
	resume: bool,
) -> tuple[int, float | None]:
	"""Load latest checkpoint state when available and resume is enabled."""
	if not resume:
		print("Resume disabled. Starting from step 0.")
		return 0, None

	if not checkpoint_path.exists():
		print("No latest checkpoint found. Starting from step 0.")
		return 0, None

	state = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(state["model"])
	optimizer.load_state_dict(state["optimizer"])
	start_step = int(state.get("step", 0)) + 1
	last_loss = state.get("loss")
	print(
		f"Resumed from {checkpoint_path} at step {start_step} "
		f"(last loss: {last_loss})"
	)
	return start_step, float(last_loss) if last_loss is not None else None


def save_checkpoint(
	model: XiaaM1,
	optimizer: torch.optim.Optimizer,
	step: int,
	loss: float,
	checkpoint_dir: Path,
) -> Path:
	"""Save step checkpoint and refresh checkpoints/latest.pt."""
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	step_checkpoint = checkpoint_dir / f"step_{step:05d}.pt"
	latest_checkpoint = checkpoint_dir / "latest.pt"

	state = {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"step": step,
		"loss": loss,
	}
	torch.save(state, step_checkpoint)
	shutil.copy2(step_checkpoint, latest_checkpoint)
	return step_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
	"""Build command-line argument parser for pretraining."""
	parser = argparse.ArgumentParser(description="Pre-train Xiaa M1")
	parser.add_argument("--max_steps", type=int, default=100000)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--seq_len", type=int, default=2048)
	parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
	return parser


def main() -> None:
	"""Run Xiaa M1 pretraining with periodic checkpointing and CSV logging."""
	args = build_arg_parser().parse_args()

	shard_dir = REPO_ROOT / "data" / "shards"
	checkpoint_dir = REPO_ROOT / "checkpoints"
	logs_dir = REPO_ROOT / "logs"
	logs_dir.mkdir(parents=True, exist_ok=True)

	cfg = XiaaConfig()
	cfg.batch_size = args.batch_size
	cfg.max_seq_len = args.seq_len

	device = get_device()
	print(f"Using device: {device}")

	model = XiaaM1(cfg).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=3e-4,
		weight_decay=0.1,
		betas=(0.9, 0.95),
	)

	sampler = ShardBatchSampler(shard_dir=shard_dir, seq_len=args.seq_len)
	latest_checkpoint = checkpoint_dir / "latest.pt"
	start_step, _ = load_checkpoint_if_available(
		model=model,
		optimizer=optimizer,
		checkpoint_path=latest_checkpoint,
		device=device,
		resume=args.resume,
	)

	if start_step >= args.max_steps:
		print(
			f"Start step {start_step} is already >= max_steps {args.max_steps}. "
			"Nothing to train."
		)
		return

	log_path = logs_dir / "train_loss.csv"
	file_exists = log_path.exists()
	with log_path.open("a", newline="", encoding="utf-8") as log_file:
		writer = csv.writer(log_file)
		if not file_exists or start_step == 0:
			writer.writerow(["step", "loss", "lr"])

		last_saved_step = start_step - 1
		for step in range(start_step, args.max_steps):
			start_time = time.perf_counter()

			lr = cosine_lr(step=step, max_steps=args.max_steps)
			for group in optimizer.param_groups:
				group["lr"] = lr

			x, y = sampler.sample_batch(batch_size=args.batch_size)
			x = x.to(device)
			y = y.to(device)

			optimizer.zero_grad(set_to_none=True)
			_, loss = model(x, y)
			if loss is None:
				raise RuntimeError("Model returned None loss while targets were provided.")

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			loss_value = float(loss.item())
			elapsed = max(time.perf_counter() - start_time, 1e-6)
			tokens_per_sec = (args.batch_size * args.seq_len) / elapsed

			print(
				f"Step {step} | Loss: {loss_value:.4f} | "
				f"LR: {lr:.6f} | Tokens/sec: {tokens_per_sec:.0f}"
			)

			writer.writerow([step, f"{loss_value:.6f}", f"{lr:.8f}"])
			log_file.flush()

			if (step + 1) % 500 == 0:
				checkpoint_path = save_checkpoint(
					model=model,
					optimizer=optimizer,
					step=step,
					loss=loss_value,
					checkpoint_dir=checkpoint_dir,
				)
				print(f"Checkpoint saved: {checkpoint_path}")
				last_saved_step = step

		if last_saved_step != args.max_steps - 1:
			checkpoint_path = save_checkpoint(
				model=model,
				optimizer=optimizer,
				step=args.max_steps - 1,
				loss=loss_value,
				checkpoint_dir=checkpoint_dir,
			)
			print(f"Final checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
	main()
