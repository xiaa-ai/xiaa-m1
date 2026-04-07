from dataclasses import dataclass

@dataclass
class XiaaConfig:
    # Model size — ~300M params
    vocab_size:   int   = 32000
    d_model:      int   = 1024
    n_heads:      int   = 16
    n_kv_heads:   int   = 4       # Grouped Query Attention
    n_layers:     int   = 24
    ffn_mult:     float = 2.6875  # SwiGLU hidden = d_model * ffn_mult
    max_seq_len:  int   = 2048
    rope_theta:   float = 10000.0
    dropout:      float = 0.0

    # Training
    batch_size:   int   = 8
    lr:           float = 3e-4
    weight_decay: float = 0.1
    grad_clip:    float = 1.0

    # For tiny test runs on Mac (swap back for real training)
    @classmethod
    def tiny(cls):
        cfg = cls()
        cfg.d_model    = 128
        cfg.n_heads    = 4
        cfg.n_kv_heads = 2
        cfg.n_layers   = 4
        cfg.vocab_size = 32000
        return cfg