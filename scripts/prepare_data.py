"""Prepare tokenizer, pretraining shards, and chat SFT data for Xiaa M1."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import numpy as np
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from xiaa.chat_format import format_chat
from xiaa.tokenizer import XiaaTokenizer, train_tokenizer


SYSTEM_PROMPT = "You are Xiaa, a helpful AI assistant made by Xiaa AI."
TOKENIZER_CORPUS_FINEWEB_DOCS = 2_000_000
TOKENIZER_CORPUS_BN_DOCS = 1_000_000
TOKENIZER_CORPUS_HI_DOCS = 1_000_000
SANGRAHA_BENGALI_LANGUAGE = "ben_Beng"
SANGRAHA_HINDI_LANGUAGE = "hin_Deva"
ENGLISH_CORPUS_REUSE_THRESHOLD_BYTES = 100 * 1024 * 1024

PRETRAIN_FINEWEB_TOKENS = 8_000_000_000
PRETRAIN_BN_TOKENS = 1_000_000_000
PRETRAIN_HI_TOKENS = 1_000_000_000
SHARD_SIZE_TOKENS = 10_000_000


def ensure_data_directories(repo_root: Path) -> tuple[Path, Path, Path]:
	"""Create all required data directories and return their paths."""
	raw_dir = repo_root / "data" / "raw"
	shards_dir = repo_root / "data" / "shards"
	chat_dir = repo_root / "data" / "chat"
	raw_dir.mkdir(parents=True, exist_ok=True)
	shards_dir.mkdir(parents=True, exist_ok=True)
	chat_dir.mkdir(parents=True, exist_ok=True)
	return raw_dir, shards_dir, chat_dir


def normalize_text(text: str) -> str:
	"""Normalize whitespace while preserving Unicode text."""
	return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def first_string(value: Any) -> str:
	"""Extract the first non-empty string from nested values."""
	if isinstance(value, str):
		return value
	if isinstance(value, dict):
		preferred_keys = (
			"text",
			"content",
			"value",
			"document",
			"article",
			"raw_text",
			"prompt",
			"response",
			"output",
		)
		for key in preferred_keys:
			if key in value:
				candidate = first_string(value[key])
				if candidate:
					return candidate
		for nested in value.values():
			candidate = first_string(nested)
			if candidate:
				return candidate
	if isinstance(value, list):
		for nested in value:
			candidate = first_string(nested)
			if candidate:
				return candidate
	return ""


def extract_record_text(record: dict[str, Any]) -> str:
	"""Extract a normalized text field from a dataset example."""
	return normalize_text(first_string(record)).strip()


def stream_fineweb() -> Iterable[dict[str, Any]]:
	"""Return a streaming iterator over FineWeb-Edu documents."""
	return load_dataset(
		"HuggingFaceFW/fineweb-edu",
		"sample-10BT",
		split="train",
		streaming=True,
	)


def extract_ultrachat_messages_text(record: dict[str, Any]) -> str:
	"""Extract flattened text from UltraChat messages for fallback corpora."""
	messages = record.get("messages")
	if isinstance(messages, list):
		chunks: list[str] = []
		for message in messages:
			if not isinstance(message, dict):
				continue
			content = normalize_text(
				str(message.get("content") or message.get("text") or message.get("value") or "")
			).strip()
			if content:
				chunks.append(content)
		if chunks:
			return " ".join(chunks)
	return extract_record_text(record)


def stream_ultrachat_indic_fallback(language: str) -> Iterable[dict[str, Any]]:
	"""Return UltraChat stream used when Sangraha is unavailable."""
	try:
		stream = load_dataset(
			"HuggingFaceH4/ultrachat_200k",
			split="train_sft",
			streaming=True,
		)
	except Exception as exc:  # pragma: no cover - depends on remote dataset availability
		print(
			"Fallback dataset load failed for "
			f"{language}: {exc}. Continuing with empty fallback stream."
		)
		return iter(())

	def generator() -> Iterator[dict[str, Any]]:
		for record in stream:
			text = extract_ultrachat_messages_text(record)
			if text:
				yield {"text": text}

	return generator()


def stream_sangraha(language: str) -> Iterable[dict[str, Any]]:
	"""Return Sangraha stream for a language, with automatic fallback."""
	try:
		return load_dataset(
			"ai4bharat/sangraha",
			language,
			split="train",
			streaming=True,
			trust_remote_code=True,
		)
	except Exception as exc:  # pragma: no cover - depends on remote dataset availability
		print(f"ai4bharat/sangraha unavailable for {language}: {exc}")
		print(
			"Falling back to HuggingFaceH4/ultrachat_200k train_sft "
			"and extracting text from messages."
		)
		return stream_ultrachat_indic_fallback(language)


def write_limited_documents(
	dataset_stream: Iterable[dict[str, Any]],
	output_file: TextIO,
	limit: int,
	source_name: str,
) -> int:
	"""Write up to limit documents into the tokenizer corpus file."""
	written = 0
	for record in dataset_stream:
		text = extract_record_text(record)
		if not text:
			continue

		output_file.write(text[:500] + "\n")
		written += 1
		if written % 100_000 == 0:
			print(f"{source_name}: wrote {written:,}/{limit:,} docs for tokenizer corpus")
		if written >= limit:
			break

	print(f"{source_name}: completed {written:,} docs for tokenizer corpus")
	return written


def prepare_tokenizer_corpus(corpus_path: Path) -> None:
	"""Build tokenizer corpus text file from multilingual sources."""
	print("Step 1/3: Preparing tokenizer corpus...")
	skip_english = (
		corpus_path.exists()
		and corpus_path.stat().st_size > ENGLISH_CORPUS_REUSE_THRESHOLD_BYTES
	)

	file_mode = "a" if skip_english else "w"
	with corpus_path.open(file_mode, encoding="utf-8") as handle:
		if skip_english:
			print(
				"Existing tokenizer corpus is >100MB. "
				"Skipping FineWeb English section and appending Indic data only."
			)
		else:
			write_limited_documents(
				dataset_stream=stream_fineweb(),
				output_file=handle,
				limit=TOKENIZER_CORPUS_FINEWEB_DOCS,
				source_name="fineweb-edu",
			)

		try:
			write_limited_documents(
				dataset_stream=stream_sangraha(SANGRAHA_BENGALI_LANGUAGE),
				output_file=handle,
				limit=TOKENIZER_CORPUS_BN_DOCS,
				source_name="sangraha-ben_Beng",
			)
			write_limited_documents(
				dataset_stream=stream_sangraha(SANGRAHA_HINDI_LANGUAGE),
				output_file=handle,
				limit=TOKENIZER_CORPUS_HI_DOCS,
				source_name="sangraha-hin_Deva",
			)
		except Exception as exc:
			print(f"Sangraha Indic section failed: {exc}")
			print(
				"Falling back to HuggingFaceH4/ultrachat_200k train_sft "
				"for Indic substitute corpus generation."
			)
			write_limited_documents(
				dataset_stream=stream_ultrachat_indic_fallback(SANGRAHA_BENGALI_LANGUAGE),
				output_file=handle,
				limit=TOKENIZER_CORPUS_BN_DOCS,
				source_name="ultrachat-fallback-ben_Beng",
			)
			write_limited_documents(
				dataset_stream=stream_ultrachat_indic_fallback(SANGRAHA_HINDI_LANGUAGE),
				output_file=handle,
				limit=TOKENIZER_CORPUS_HI_DOCS,
				source_name="ultrachat-fallback-hin_Deva",
			)
	print(f"Tokenizer corpus saved: {corpus_path}")


class ShardWriter:
	"""Buffered uint16 shard writer for tokenized pretraining data."""

	def __init__(self, shard_dir: Path, shard_size_tokens: int = SHARD_SIZE_TOKENS) -> None:
		"""Initialize shard writer state."""
		self.shard_dir = shard_dir
		self.shard_size_tokens = shard_size_tokens
		self.current = np.empty(self.shard_size_tokens, dtype=np.uint16)
		self.filled = 0
		self.shard_index = 0
		self.total_tokens = 0
		self.next_progress_mark = 100_000_000

	def write(self, token_ids: list[int]) -> None:
		"""Write token IDs into fixed-size shard files."""
		if not token_ids:
			return

		array = np.asarray(token_ids, dtype=np.uint16)
		cursor = 0
		while cursor < array.size:
			available = self.shard_size_tokens - self.filled
			take = min(available, int(array.size - cursor))

			self.current[self.filled : self.filled + take] = array[cursor : cursor + take]
			self.filled += take
			cursor += take
			self.total_tokens += take

			while self.total_tokens >= self.next_progress_mark:
				print(f"Tokenization progress: {self.next_progress_mark:,} tokens")
				self.next_progress_mark += 100_000_000

			if self.filled == self.shard_size_tokens:
				self.flush_current(full_shard=True)

	def flush_current(self, full_shard: bool) -> None:
		"""Flush the currently buffered tokens to a new shard file."""
		if self.filled == 0:
			return

		shard_path = self.shard_dir / f"shard_{self.shard_index:04d}.bin"
		if full_shard:
			self.current.tofile(shard_path)
		else:
			self.current[: self.filled].tofile(shard_path)

		print(
			f"Saved shard {self.shard_index:04d}: {self.filled:,} tokens -> {shard_path.name}"
		)
		self.shard_index += 1
		self.filled = 0

	def finalize(self) -> None:
		"""Flush any remaining tokens as a final partial shard."""
		self.flush_current(full_shard=False)


def iter_tokenized_documents(
	dataset_stream: Iterable[dict[str, Any]],
	tokenizer: XiaaTokenizer,
	token_budget: int,
	source_name: str,
) -> Iterator[list[int]]:
	"""Yield tokenized documents until the source token budget is reached."""
	emitted = 0
	for record in dataset_stream:
		if emitted >= token_budget:
			break

		text = extract_record_text(record)
		if not text:
			continue

		token_ids = tokenizer.encode(text, add_bos=False, add_eos=True)
		if not token_ids:
			continue

		remaining = token_budget - emitted
		if len(token_ids) > remaining:
			token_ids = token_ids[:remaining]

		emitted += len(token_ids)
		yield token_ids

	if emitted < token_budget:
		print(
			f"Warning: {source_name} ended early at {emitted:,}/{token_budget:,} tokens"
		)
	else:
		print(f"{source_name}: reached token budget {token_budget:,}")


def clear_existing_shards(shard_dir: Path) -> None:
	"""Remove old shard files before writing a new shard set."""
	removed = 0
	for shard_path in shard_dir.glob("shard_*.bin"):
		shard_path.unlink()
		removed += 1
	if removed:
		print(f"Removed {removed} existing shard files")


def prepare_pretraining_shards(shard_dir: Path, tokenizer: XiaaTokenizer) -> None:
	"""Create uint16 binary training shards from multilingual streams."""
	print("Step 2/3: Building pretraining shards...")
	clear_existing_shards(shard_dir)
	writer = ShardWriter(shard_dir=shard_dir, shard_size_tokens=SHARD_SIZE_TOKENS)

	sources: list[tuple[str, Iterable[dict[str, Any]], int]] = [
		("fineweb-edu", stream_fineweb(), PRETRAIN_FINEWEB_TOKENS),
	]

	try:
		sources.extend(
			[
				(
					"sangraha-ben_Beng",
					stream_sangraha(SANGRAHA_BENGALI_LANGUAGE),
					PRETRAIN_BN_TOKENS,
				),
				(
					"sangraha-hin_Deva",
					stream_sangraha(SANGRAHA_HINDI_LANGUAGE),
					PRETRAIN_HI_TOKENS,
				),
			]
		)
	except Exception as exc:
		print(f"Sangraha pretraining stream setup failed: {exc}")
		print(
			"Falling back to HuggingFaceH4/ultrachat_200k train_sft "
			"for Indic substitute pretraining streams."
		)
		sources.extend(
			[
				(
					"ultrachat-fallback-ben_Beng",
					stream_ultrachat_indic_fallback(SANGRAHA_BENGALI_LANGUAGE),
					PRETRAIN_BN_TOKENS,
				),
				(
					"ultrachat-fallback-hin_Deva",
					stream_ultrachat_indic_fallback(SANGRAHA_HINDI_LANGUAGE),
					PRETRAIN_HI_TOKENS,
				),
			]
		)

	for source_name, stream, token_budget in sources:
		print(f"Tokenizing source: {source_name} (target {token_budget:,} tokens)")
		for token_ids in iter_tokenized_documents(stream, tokenizer, token_budget, source_name):
			writer.write(token_ids)

	writer.finalize()
	print(
		f"Pretraining shards complete: {writer.shard_index:,} shards, "
		f"{writer.total_tokens:,} total tokens"
	)


def extract_ultrachat_pair(example: dict[str, Any]) -> tuple[str, str] | None:
	"""Extract a user/assistant pair from an UltraChat example."""
	messages = example.get("messages")
	if isinstance(messages, list):
		user_message = ""
		assistant_message = ""
		for message in messages:
			if not isinstance(message, dict):
				continue
			role = str(message.get("role") or message.get("from") or "").lower().strip()
			content = normalize_text(
				str(message.get("content") or message.get("text") or message.get("value") or "")
			).strip()
			if role == "user" and content:
				user_message = content
			if role == "assistant" and content and user_message:
				assistant_message = content
				break
		if user_message and assistant_message:
			return user_message, assistant_message

	prompt = normalize_text(str(example.get("prompt") or "")).strip()
	response = normalize_text(
		str(example.get("response") or example.get("answer") or "")
	).strip()
	if prompt and response:
		return prompt, response

	return None


def extract_bengali_alpaca_pair(example: dict[str, Any]) -> tuple[str, str] | None:
	"""Extract a user/assistant pair from Alpaca-style Bengali data."""
	instruction = normalize_text(str(example.get("instruction") or "")).strip()
	input_text = normalize_text(str(example.get("input") or "")).strip()
	assistant = normalize_text(
		str(example.get("output") or example.get("response") or "")
	).strip()

	if not assistant:
		return None

	if instruction and input_text:
		user_text = f"{instruction}\n{input_text}"
	else:
		user_text = instruction or input_text

	if not user_text:
		return None
	return user_text, assistant


def build_chatml_example(user_message: str, assistant_message: str) -> str:
	"""Create one ChatML-formatted training sample."""
	return format_chat(
		[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": user_message},
			{"role": "assistant", "content": assistant_message},
		]
	)


def prepare_chat_sft_data(chat_output_path: Path) -> None:
	"""Create ChatML JSONL data for supervised fine-tuning."""
	print("Step 3/3: Preparing chat SFT data...")
	total_saved = 0

	with chat_output_path.open("w", encoding="utf-8") as handle:
		ultrachat_stream = load_dataset(
			"HuggingFaceH4/ultrachat_200k",
			split="train_sft",
			streaming=True,
		)

		for example in itertools.islice(ultrachat_stream, 50_000):
			pair = extract_ultrachat_pair(example)
			if pair is None:
				continue

			user_text, assistant_text = pair
			payload = {"text": build_chatml_example(user_text, assistant_text)}
			handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
			total_saved += 1

		print(f"ultrachat_200k: saved {total_saved:,} examples")

		bengali_saved = 0
		try:
			bengali_dataset = load_dataset("iamshnoo/alpaca-cleaned-bengali", split="train")
			for example in bengali_dataset:
				pair = extract_bengali_alpaca_pair(example)
				if pair is None:
					continue

				user_text, assistant_text = pair
				payload = {"text": build_chatml_example(user_text, assistant_text)}
				handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
				total_saved += 1
				bengali_saved += 1
			print(f"alpaca-cleaned-bengali: saved {bengali_saved:,} examples")
		except Exception as exc:  # pragma: no cover - depends on remote dataset availability
			print(f"alpaca-cleaned-bengali unavailable, skipping gracefully: {exc}")

	print(f"Total chat SFT examples saved: {total_saved:,}")
	print(f"SFT data path: {chat_output_path}")


def main() -> None:
	"""Run the full Xiaa M1 data preparation pipeline."""
	raw_dir, shards_dir, chat_dir = ensure_data_directories(REPO_ROOT)
	tokenizer_corpus_path = raw_dir / "tokenizer_corpus.txt"
	tokenizer_prefix = REPO_ROOT / "xiaa" / "xiaa_tokenizer"
	chat_jsonl_path = chat_dir / "sft_data.jsonl"

	prepare_tokenizer_corpus(tokenizer_corpus_path)
	tokenizer_model_path = train_tokenizer(
		input_file=tokenizer_corpus_path,
		model_prefix=tokenizer_prefix,
		vocab_size=32000,
	)
	print(f"Tokenizer model trained: {tokenizer_model_path}")

	tokenizer = XiaaTokenizer(tokenizer_model_path)
	prepare_pretraining_shards(shards_dir, tokenizer)
	prepare_chat_sft_data(chat_jsonl_path)
	print("Data preparation complete.")


if __name__ == "__main__":
	main()
