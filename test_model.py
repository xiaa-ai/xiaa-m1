import torch
from xiaa.model import XiaaM1
from xiaa.config import XiaaConfig

# Use tiny config for quick test
cfg = XiaaConfig.tiny()
model = XiaaM1(cfg)

print(f"Xiaa M1 (tiny) — {model.num_params()/1e6:.1f}M params")

# Check MPS (Apple Silicon GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using: {device}")
model = model.to(device)

# Fake batch
x = torch.randint(0, cfg.vocab_size, (2, 64)).to(device)
y = torch.randint(0, cfg.vocab_size, (2, 64)).to(device)

logits, loss = model(x, targets=y)
print(f"Logits shape: {logits.shape}")   # (2, 64, 32000)
print(f"Loss: {loss.item():.4f}")        # should be ~10.4 (= log(32000))
print("✅ Xiaa M1 architecture working")

# Full 300M config
cfg_full = XiaaConfig()
model_full = XiaaM1(cfg_full)
print(f"\nXiaa M1 (full) — {model_full.num_params()/1e6:.0f}M params")