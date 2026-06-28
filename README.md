# infr

Pure-Rust LLM inference engine. Vulkan-first, built to run on any mainstream
GPU.

> Early WIP. The only non-Rust parts are the GPU driver calls (Vulkan via `ash`)
> and the compute shaders (SPIR-V).

## Goal

A from-the-metal inference server that works across AMD / NVIDIA / Intel
(Vulkan) and Apple (MoltenVK), with native backends addable later behind a
`Compute` trait.

## Status

Runs **Llama / Qwen2 / Qwen3** (dense) on the Vulkan backend, competitive with
llama.cpp at long context (`infr compare`). **Qwen3.5 / Qwen3.6** (`qwen35` /
Qwen3-Next — hybrid gated-DeltaNet + attention) run via a CPU reference
(`docs/QWEN35.md`); a Vulkan/hybrid path is planned. DiffusionGemma (the
original target) is future work.

```bash
infr pull   <model-ref>        # hf:org/repo[:file] | ollama:name[:tag] | path
infr run    <model-ref> [msg]  # terminal chat (auto-pulls)
infr serve  <model-ref>        # OpenAI-compatible HTTP API
infr bench / infr compare      # tok/s benchmarks vs llama.cpp
```

## Scope

- **Format:** GGUF
- **Models:** Llama / Qwen3 dense (GPU); Qwen3.5/3.6 (CPU ref); DiffusionGemma
  (planned)
- **GPU:** AMD / NVIDIA / Intel via Vulkan (cooperative-matrix matmul)
- **Store:** own cache at `$XDG_CACHE_HOME/infr/models` (standalone HF + Ollama
  HTTP pulls)
- **API:** OpenAI-compatible HTTP (streaming) — works with opencode / Claude
  Code CLI

## Architecture

```
server   axum + SSE  ->  OpenAI /v1
decode   DecodeStrategy   (AutoRegressive; DiffusionDenoise later)
model    Model            (Llama/Qwen3; Qwen3-Next CPU ref; DiffusionGemma later)
runtime  tensors, KV cache, command/descriptor management
loader   WeightSource     (Gguf; safetensors later)
compute  Compute          (Vulkan via ash + SPIR-V; Metal/CUDA later)
```

## License

[MIT](LICENSE)
