# DeepSeek-V3 @ ~100M parameters — sizing blueprint

This is a *reference card* the skill reads before implementing. The numbers below are a starting point — verify the param count at init and adjust if off-target by >30%.

## Provenance

Architecture: DeepSeek-V3 (paper: arXiv 2412.19437). Two distinctive components:
- **MLA** — Multi-head Latent Attention (low-rank Q/K/V joint compression + decoupled RoPE).
- **DeepSeekMoE** — fine-grained experts with a shared expert and top-K routing.

Reference impl: `https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main` (look at `modeling_deepseek.py`, `configuration_deepseek.py`).

## Target config (~100M params)

```python
config = dict(
    vocab_size      = 32000,    # use deepseek-ai/DeepSeek-V3-Base tokenizer if reachable; fall back to gpt2
    hidden_size     = 512,      # d_model
    num_hidden_layers = 12,
    num_attention_heads = 8,
    max_position_embeddings = 512,

    # MLA
    q_lora_rank        = 192,
    kv_lora_rank       = 128,
    qk_nope_head_dim   = 32,
    qk_rope_head_dim   = 32,
    v_head_dim         = 64,
    rope_theta         = 10000.0,

    # MoE — only apply to layers >= first_k_dense_replace (e.g. 1)
    first_k_dense_replace = 1,
    n_routed_experts   = 4,
    n_shared_experts   = 1,
    num_experts_per_tok = 2,
    moe_intermediate_size = 1024,   # per-expert FFN hidden
    intermediate_size  = 1408,      # dense FFN hidden (for first_k_dense_replace dense layers)

    rms_norm_eps       = 1e-6,
    tie_word_embeddings = True,
)
```

Param budget (back-of-envelope, total — not active):
- Token embedding: 32000 × 512 ≈ 16.4M (tied to LM head, so counted once)
- 11 MoE layers × (MLA ≈ 1.5M + 4 routed experts × 512·1024·2 + shared expert) ≈ 7–8M each → ≈ 80M
- 1 dense layer + norms: ≈ 3M
- **Total ≈ 95–110M**

Active params per token (top-2 of 4 + shared): roughly half of MoE FFN weight → ~50M active.

## Tokenizer

Try in this order:
1. `AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")` — may need `trust_remote_code=True` and HF auth.
2. `AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")` — same family.
3. Fallback: `AutoTokenizer.from_pretrained("gpt2")` — vocab=50257, set `vocab_size=50257` in the config above.

## Tiny dataset

Pick whichever loads fastest on eva02:
- `roneneldan/TinyStories` (`split="train", streaming=True`) — clean, small, good loss signal.
- `karpathy/tiny_shakespeare` — local, ~1MB.
- `wikitext` (`wikitext-2-raw-v1`) — slightly larger.

Tokenize to seq_len=256, batch=8 → 2048 tokens/step.

## Training recipe (100-step smoke train)

```
optimizer  : AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
schedule   : linear warmup 50 steps → constant
precision  : bf16 (A6000 supports it natively)
batch      : 8 × seq 256
grad_clip  : 1.0
steps      : 100
log_every  : 1   # every step into train.log as `step=N loss=<val>`
ckpt       : at step 100 → ckpt.pt
```

## NaN guard (must be in train.py)

```python
if not torch.isfinite(loss):
    optimizer.zero_grad(set_to_none=True)
    for g in optimizer.param_groups:
        g["lr"] *= 0.5
    nan_streak += 1
    if nan_streak >= 5:
        raise RuntimeError(f"NaN persisted at step {step}")
    continue
nan_streak = 0
```

## Pass criteria (skill done conditions)

- Model class instantiates and prints `total_params: <80–130M>` at init.
- One forward pass on `torch.randint(0, vocab, (1, 64))` returns logits of shape `(1, 64, vocab)` without crashing.
- `train.log` has ≥100 lines `step=N loss=<finite>`.
- `loss[99] < loss[0]`.
- `ckpt.pt` exists.

## Common pitfalls

- **MLA RoPE split**: only the `qk_rope_head_dim` portion of Q and K gets RoPE — the `qk_nope_head_dim` portion stays untouched. Easy to apply RoPE to the whole head and silently break the model.
- **MoE load balancing**: with only 4 experts, expert collapse is common. Add an aux load-balance loss term (`coef=0.01`), or skip it for the 100-step smoke run.
- **Tied embeddings**: don't double-count when reporting param count; use `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- **bf16 + grad**: keep optimizer state in fp32; PyTorch's AdamW handles this if you call `model.to(torch.bfloat16)` only on params and use `torch.autocast`. Easier: full bf16 model, no autocast, AdamW will keep moments in fp32 internally if you use `foreach=True` (default).
