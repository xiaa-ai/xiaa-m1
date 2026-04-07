"""SentencePiece tokenizer helpers for Xiaa M1."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import sentencepiece as spm


SPECIAL_TOKENS: tuple[str, ...] = (
	"<|system|>",
	"<|user|>",
	"<|assistant|>",
	"<|end|>",
)


def train_tokenizer(
	input_file: str | Path,
	model_prefix: str | Path,
	vocab_size: int = 32000,
) -> Path:
	"""Train a SentencePiece BPE tokenizer and return the model path."""
	input_path = Path(input_file)
	model_prefix_path = Path(model_prefix)

	if not input_path.exists():
		raise FileNotFoundError(f"Tokenizer corpus not found: {input_path}")

	model_prefix_path.parent.mkdir(parents=True, exist_ok=True)
	spm.SentencePieceTrainer.train(
		input=str(input_path),
		model_prefix=str(model_prefix_path),
		vocab_size=vocab_size,
		character_coverage=0.9995,
		bos_id=2,
		eos_id=3,
		pad_id=0,
		unk_id=1,
		model_type="bpe",
		input_sentence_size=5_000_000,
		shuffle_input_sentence=True,
		user_defined_symbols=",".join(SPECIAL_TOKENS),
	)
	return model_prefix_path.with_suffix(".model")


class XiaaTokenizer:
	"""Thin wrapper around SentencePiece for Xiaa M1 tokenization."""

	def __init__(self, model_path: str | Path | None = None) -> None:
		"""Load the SentencePiece model from disk."""
		if model_path is None:
			model_path = Path(__file__).resolve().parent / "xiaa_tokenizer.model"

		self.model_path = Path(model_path)
		if not self.model_path.exists():
			raise FileNotFoundError(
				"Tokenizer model file not found. "
				f"Expected: {self.model_path}. Run scripts/prepare_data.py first."
			)
		self.sp_model = spm.SentencePieceProcessor(model_file=str(self.model_path))

	def encode(
		self,
		text: str,
		add_bos: bool = True,
		add_eos: bool = True,
	) -> list[int]:
		"""Encode text into token IDs."""
		return self.sp_model.encode(
			text,
			out_type=int,
			add_bos=add_bos,
			add_eos=add_eos,
		)

	def decode(self, ids: Sequence[int]) -> str:
		"""Decode token IDs back to text."""
		return self.sp_model.decode(list(ids))

	def token_to_id(self, token: str) -> int:
		"""Return the token ID for a given piece."""
		return int(self.sp_model.piece_to_id(token))

	def id_to_token(self, token_id: int) -> str:
		"""Return the token piece for a given token ID."""
		return str(self.sp_model.id_to_piece(token_id))

	def __len__(self) -> int:
		"""Return the vocabulary size."""
		return int(self.sp_model.get_piece_size())
