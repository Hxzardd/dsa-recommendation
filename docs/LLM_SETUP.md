# LLM setup — Ollama (dev) · vLLM in WSL2 · hosted vLLM (prod)

The service talks to **any OpenAI-compatible `/v1/chat/completions` endpoint**
(`app/llm/client.py`). "vLLM" is the production target, but the client is just
speaking the OpenAI protocol — so in development you can point it at Ollama, and
in production at a real vLLM server, **without any code change**. Only two env
vars move: `VLLM_BASE_URL` and `VLLM_MODEL`.

```
VLLM_BASE_URL   → the base; the client calls {base}/v1/chat/completions
VLLM_MODEL      → the served model name
VLLM_API_KEY    → optional bearer, if the endpoint requires one
```

If the endpoint is unreachable the service **degrades gracefully** (rule-only /
`confidence: 0`) — it never 5xxes. See [ARCHITECTURE.md](ARCHITECTURE.md#graceful-degradation).

---

## Why not "just pip install vllm" on Windows?

vLLM is **Linux + NVIDIA-GPU** software — there is **no native Windows build**.
To run it on Windows you need **WSL2** (or Docker with the WSL2 GPU backend) plus
an NVIDIA GPU. And model size is capped by **VRAM**, not system RAM.

### VRAM sizing (rough)

| Model | fp16 weights | 4-bit (AWQ/GPTQ) | Fits on… |
|---|---|---|---|
| 7B  | ~14 GB | ~5–6 GB | 8–12 GB card (quantized) |
| 14B | ~28 GB | ~9–10 GB | 12–16 GB card (quantized, tight) |
| 32B | ~64 GB | ~18–20 GB | 24 GB+ card or multi-GPU / cloud |

> On a 12 GB card the practical ceiling is a **7B model, 4-bit quantized**. A 7B
> model on vLLM is the **same tier** as `qwen2.5-coder:7b` on Ollama — locally,
> vLLM buys you serving throughput, not better analysis. The real quality jump
> (32B) needs a bigger/cloud GPU.

---

## 1) Ollama — fastest local dev (recommended when you already have it)

Zero vLLM install. Ollama serves the OpenAI API at `http://localhost:11434/v1`.

```bash
ollama serve                       # or the Ollama app
ollama pull qwen2.5-coder:7b
```

`.env`:
```bash
VLLM_BASE_URL=http://localhost:11434
VLLM_MODEL=qwen2.5-coder:7b
# VLLM_API_KEY=                     # leave blank for Ollama
```

Good for wiring and end-to-end testing. Quality is the small-model tier — use a
real vLLM/32B for production.

## 2) vLLM in WSL2 — local, needs an NVIDIA GPU

Example sizing below is for an **RTX 4070 (12 GB desktop / 8 GB laptop)** → a
**7B AWQ** model.

```powershell
# PowerShell (admin): installs WSL2 + Ubuntu, then reboot
wsl --install
```
Keep your **Windows NVIDIA driver current** — it ships WSL CUDA. Do **not**
install a separate driver inside Ubuntu. Then, inside Ubuntu:

```bash
sudo apt update && sudo apt install -y python3.12-venv
python3.12 -m venv ~/vllm && source ~/vllm/bin/activate
pip install vllm

# 7B code model, 4-bit AWQ. Port 8001 (the ML service uses 8000).
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --quantization awq \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

If it OOMs (or on an **8 GB laptop 4070**): lower `--max-model-len` to `4096`
(or `2048`) and `--gpu-memory-utilization` to `0.85`.

`.env` (WSL2 forwards `localhost`, so Windows reaches it directly):
```bash
VLLM_BASE_URL=http://localhost:8001
VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
```

## 3) Hosted vLLM — production quality (32B)

Run vLLM on a cloud GPU (RunPod, Modal, Lambda, Vast, …) or any managed
OpenAI-compatible endpoint, then point the env at it:
```bash
VLLM_BASE_URL=https://your-vllm-host          # {base}/v1/chat/completions
VLLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
VLLM_API_KEY=sk-...                            # if the host requires one
```
Nothing in the code changes.

---

## Verify any endpoint

```bash
curl "$VLLM_BASE_URL/v1/chat/completions" -H "Content-Type: application/json" \
  ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} \
  -d '{"model":"'"$VLLM_MODEL"'","messages":[{"role":"user","content":"ping"}]}'
```
A `choices[0].message.content` in the response means the service will work. Then:
```bash
uv run uvicorn app.main:app --port 8099
```

## Recommendation

- **Develop** against Ollama (identical quality to a local 7B, zero setup), or
  vLLM-in-WSL2 when you specifically want to exercise the vLLM serving path.
- **Deploy** the 32B on a cloud GPU where vLLM earns its keep — same env knobs.
