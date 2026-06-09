# Rotating the Week

**[rotating-week.vercel.app](https://rotating-week.vercel.app)** — an interactive, fully in-browser recreation of the day-of-week *circular feature* in GPT-2 small, from Engels et al., [*Not All Language Model Features Are Linear*](https://arxiv.org/abs/2405.14860) (2024).

Drag a dial to **rotate GPT-2's residual stream around the recovered day-of-week circle**, and the model's next-token prediction walks around the week in lockstep — rotate from Tuesday by two days and it predicts Thursday. Real GPT-2 forward passes, client-side via ONNX Runtime Web. No server, no GPU, no API key.

## The claim being tested

GPT-2 small represents the days of the week on a *circle* in its residual stream and computes succession by *rotating around it*. This is a **causal** claim, not a visualization: rotating only the in-plane component by `k` days advances the predicted next-token day by `k`. Additively steering a "Wednesday direction" would not show this — any linear feature would do that. The result is the rotation.

## What actually happens in your browser

- GPT-2 small is split at `blocks.11.hook_resid_pre` into two ONNX graphs:
  - **encoder** — embeddings + blocks 0–10 → the residual stream at that point
  - **decoder** — block 11 + final layernorm + unembedding → logits
- Both run client-side via ONNX Runtime Web. The encoder runs once per prompt; the decoder re-runs on every dial-turn.
- The rotation, projection, and ablation are plain JS on the residual *between* the two halves.
- The circle basis `(u1, u2)`, its center, and the seven *measured* day angles were recovered offline with TransformerLens and baked in as constants — the browser does not recompute them, but every model forward pass is live.

## Validation — the part that matters

"It ran" is not "it's correct." Every step has a number:

| check | result |
|---|---|
| split round-trip (full GPT-2 vs encoder→decoder) | **cosine 0.99999**, max-abs-diff ~3e-5 |
| in-browser rotation vs TransformerLens patched hook | same tolerance |
| **sufficiency** — rotation → succession lockstep | **6/7 days** (Sun→Mon weekend wrap is the honest miss) |
| **necessity** — ablate the day-circle subspace | removes **~56%** of the successor probability |
| necessity control — ablate a matched random plane | removes **~0%** |

Sufficiency (you can steer it) **and** necessity (remove it and it breaks, but a random subspace of the same size doesn't) together make the full causal argument.

## Honest scope

This **reproduces and makes interactive a published finding — it is not a new result.** Two caveats worth stating plainly, both surfaced in the app:

- The circle is geometrically present at *every* layer but only becomes *causally* load-bearing in the last couple — the geometrically cleanest layer is causally inert.
- The effect is real but **partial**, and noticeably cleaner for days (7-cycle) than months (12-cycle) in GPT-2 small.

## Run it locally

```bash
pnpm install
pnpm dev
```

The two ONNX halves (~655 MB, fp32) are fetched at runtime from the [Hugging Face Hub](https://huggingface.co/lapiskasha/gpt2-circular-steering) — they're too large to bundle. Cross-origin isolation (COOP/COEP) is set in `vite.config.ts` so the threaded WASM works.

## Stack

Frontend: React + Vite + ONNX Runtime Web. Offline analysis/export: TransformerLens + PyTorch + onnx/onnxruntime.

## Credit

Method and finding: Engels, Michaud, Liao, Gurnee, Tegmark (2024). This interactive reproduction built by [Akash Patel](https://www.akashpa.tel).
