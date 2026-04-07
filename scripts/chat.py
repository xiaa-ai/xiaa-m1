"""Interactive CLI chat with a fine-tuned Xiaa M1 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xiaa.chat_format import format_chat, parse_chat
from xiaa.config import XiaaConfig
from xiaa.model import XiaaM1
from xiaa.tokenizer import XiaaTokenizer


SYSTEM_PROMPT = "You are Xiaa, a helpful AI assistant made by Xiaa AI."


def get_device() -> torch.device:
    """Select inference device with mps > cuda > cpu priority."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Sample a token using temperature, top-k, and nucleus filtering."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        kth = values[:, [-1]]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, -float("inf")), scaled)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)

        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
        scaled = torch.full_like(scaled, -float("inf"))
        scaled.scatter_(1, sorted_indices, sorted_logits)

    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def stream_response(
    model: XiaaM1,
    tokenizer: XiaaTokenizer,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[int]:
    """Generate and stream assistant response tokens to terminal."""
    model.eval()
    generated: list[int] = []

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    end_token_id = tokenizer.token_to_id("<|end|>")

    print("Xiaa: ", end="", flush=True)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = input_ids[:, -model.cfg.max_seq_len :]
            logits, _ = model(window)
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature, top_k, top_p)
            token_id = int(next_token.item())

            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated.append(token_id)

            if token_id == end_token_id:
                break

            piece = tokenizer.decode([token_id])
            print(piece, end="", flush=True)

    print()
    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for chat CLI."""
    parser = argparse.ArgumentParser(description="Chat with Xiaa M1")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/xiaa_m1_sft.pt",
        help="Path to the SFT checkpoint",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser


def main() -> None:
    """Start an interactive ChatML conversation loop in the terminal."""
    args = build_arg_parser().parse_args()

    checkpoint_path = REPO_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: {checkpoint_path}. Run scripts/finetune_chat.py first."
        )

    tokenizer = XiaaTokenizer(REPO_ROOT / "xiaa" / "xiaa_tokenizer.model")

    device = get_device()
    print(f"Using device: {device}")

    state = torch.load(checkpoint_path, map_location=device)
    cfg_dict = state.get("config")
    cfg = XiaaConfig(**cfg_dict) if isinstance(cfg_dict, dict) else XiaaConfig()

    model = XiaaM1(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Type /reset to clear conversation and /exit to quit.")
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue

        if user_text == "/exit":
            break

        if user_text == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_text})

        prompt_text = format_chat(messages) + "\n<|assistant|>\n"
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)

        response_token_ids = stream_response(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        assistant_text = tokenizer.decode(
            [tid for tid in response_token_ids if tokenizer.id_to_token(tid) != "<|end|>"]
        ).strip()
        if not assistant_text:
            assistant_text = "..."
        messages.append({"role": "assistant", "content": assistant_text})

        # Keep parser usage explicit to ensure the format remains valid ChatML over time.
        _ = parse_chat(format_chat(messages))


if __name__ == "__main__":
    main()
