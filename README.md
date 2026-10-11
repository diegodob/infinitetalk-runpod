# InfiniteTalk on RunPod

A ready-to-run container image for [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
(Wan 2.1 + audio-driven talking video): the repo, its dependencies with pinned
versions, and the compatibility patches needed to actually run it on a modern
PyTorch 2.8 / CUDA 12.8 / Python 3.12 stack.

Image: `ghcr.io/diegodob/infinitetalk-runpod:latest`

## What's inside

- Base: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (x86_64, same arch as RunPod pods)
- A venv at `/opt/venv` with `diffusers==0.31.0`, `transformers==4.51.3`,
  `accelerate==0.34.2`, `xformers==0.0.32.post1`, `optimum-quanto`, `decord`,
  `librosa`, `moviepy`, `scenedetect` and the rest of the list in `constraints.txt`
- InfiniteTalk at commit `50aa0a94` under `/opt/italk`
- The four patches (`patches/apply_patches.py`) that make it import and run

The model weights (~38 GB) are **not** in the image: run `scripts/setup_weights.sh`
once per pod (~5 minutes) or keep them on a RunPod network volume.

## Quick start

Create a pod with:

- **Image:** `ghcr.io/diegodob/infinitetalk-runpod:latest`
- **GPU:** 16 GB is the floor (see timings below), 40 GB+ is much better
- **Container disk:** 120 GB (the weights live there unless you use a volume)
- **Expose:** nothing needed; you work over SSH or the web terminal

Then, inside the pod:

```bash
bash /opt/italk/../scripts/setup_weights.sh        # if the scripts are baked in
# or, from a checkout:
# bash scripts/setup_weights.sh /workspace/weights

bash scripts/generate.sh /workspace/foto.png /workspace/audio.wav /workspace/out.mp4 20
```

`generate.sh` writes the `in.json` for you and runs the proven command line.

## Measured timings

Real numbers from a 5-second clip (81 frames at 480p, `--use_teacache`):

| GPU | VRAM | `--num_persistent_param_in_dit` | Per sampling step | 20 steps, 5 s clip |
|---|---|---|---|---|
| RTX A4000 | 16 GB | `0` (streams from RAM) | ~7 min | ~2 h 20 |
| A6000 / 48 GB class | 48 GB | larger, e.g. 400 | minutes | under an hour |

Rule of thumb: the step time scales with the number of frames, so a 10-second
clip costs roughly twice a 5-second one on the same card. On a 16 GB card a
10-second clip is ~5 hours — use a bigger card for anything longer than a few
seconds.

## Gotchas (all handled by the image)

These are the failure modes patched by `patches/apply_patches.py`:

1. `from inspect import ArgSpec` — removed in Python 3.11+
2. `no_init_weights` / `ContextManagers` — no longer exported by diffusers 0.31
3. `from kokoro import KPipeline` — optional TTS dependency, breaks the import
4. `flash_attention()` called without flash-attn installed — falls back to PyTorch SDPA
5. `xfuser` missing — stubbed, only needed for multi-GPU

Two command-line traps that are **not** patches, keep them in mind:

- `--quant_dir` wants the **file** (`.../quant_models/infinitetalk_single_fp8.safetensors`),
  not the directory, even though the help text says "directory".
- Do **not** pass `--t5_cpu`: with that flag the text encoder is called per prompt and
  the model receives a nested list, dying with
  `AttributeError: 'list' object has no attribute 'dtype'`.
- `t5_fp8.safetensors` must sit one level **above** `quant_models/`, because the code
  resolves it as `dirname(quant_dir)/t5_fp8.safetensors`. The setup script creates the
  symlink.

Also: never let `pip` upgrade `transformers` (it goes to 5.x and the diffusers import
breaks) and keep `numpy<2`. Both are pinned in `constraints.txt`.

## Building it yourself

Push to `main` and the GitHub Action builds and publishes the image to ghcr.io.
The image is large (~16 GB), so the workflow first frees disk space on the runner.

## Credits

- InfiniteTalk: [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
- Wan 2.1 base model: [Wan-AI](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
- This packaging and the compatibility patches: MIT (see `LICENSE`).
  The upstream models and code keep their own licenses (Apache-2.0 for Wan).
