# ovms-rig

A declarative loader for [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server). From a single YAML file it brings up OVMS with a validated speculative-decoding setup on a fresh machine -- no Docker, no WSL.

## What it is

One `ovms.yaml` describes:

- **`repository`** -- which models are needed (HuggingFace identity + task).
- **`models`** -- which endpoints to bring up and with which parameters (`LLMCalculatorOptions`: device, KV-cache, draft, etc.).
- **`profiles`** -- groups of endpoints you switch between with one command. Exactly one is active at a time.

The CLI expands the declaration into the files OVMS needs (`config.json`, `graph.<name>.pbtxt`) and runs the server in the foreground.

## Why

As part of a larger project, I was looking for an LLM inference runtime on Intel Arc iGPU + CPU + NPU that runs natively on Windows and gives a tangible speedup via speculative decoding. After ruling out the alternatives, OVMS was the only one standing. On a `Qwen3-14B-int8` (GPU) + `Qwen3-0.6B-int8` (CPU draft, n=7) configuration I got **11.7 tok/s** vs. **4.93 tok/s** for the no-speculation baseline -- about 2.4x.

That configuration already existed, but it was hardcoded into absolute Windows paths inside `config.json` and `graph.pbtxt`. Not reproducible, not portable, not shareable. `ovms-rig` turns it into a portable declaration while letting OVMS handle everything it already does well.

For the "why exactly this way" details, see [DECISIONS.md](DECISIONS.md).

## Installation

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and an installed OVMS binary.

```bash
uv tool install .
```

Once installed, `ovms-rig` is available as a command. The utility does **not** install the OVMS binary -- it locates it by priority: `--ovms-path` flag -> `runtime.ovms_path` from `local.yaml` -> `PATH` -> error with guidance.

## Quick start

```bash
# 1. Copy the example configs
cp config/ovms.example.yaml  config/ovms.yaml
cp config/local.example.yaml config/local.yaml

# 2. Open local.yaml and set:
#    - models.repository_path  (required: where to store models)
#    - runtime.ovms_path       (if ovms is not on PATH)
#    - runtime.cache_dir       (optional: where to keep the OpenVINO compile cache)

# 3. Pull models from the repository:
ovms-rig fetch qwen3-14b
ovms-rig fetch qwen3-0.6b   # for the draft, if you use one

# 4. Activate a profile (sets active: true on it in ovms.yaml, clears the flag on all others):
ovms-rig activate default

# 5. Run the server in the foreground:
ovms-rig start
```

Any change to `ovms.yaml` requires another `ovms-rig activate` -- it re-reads the declaration and re-renders `config.json` and the sibling `graph.pbtxt` copies. Without an argument `activate` re-applies the currently active profile.

## Adding a new model

The example `ovms.yaml` already declares `qwen3-14b` and `qwen3-0.6b`, so the Quick start works out of the box. For any other model, the full cycle is:

1. Add an entry to `repository:` in `ovms.yaml` with your own short name and the HF identity:
   ```yaml
   repository:
     my-model:
       hf: OpenVINO/Qwen3-8B-int8-ov
       task: text_generation
   ```
2. `ovms-rig fetch my-model` -- pulls the model to disk.
3. Add a `models:` entry that references it via `.source: my-model`, with the `graph:` parameters you want.
4. Reference the new endpoint from a profile, then `ovms-rig activate <profile>`.

`remove` is the inverse of `fetch`: deletes the on-disk artifacts and the corresponding `config.json` entry, but does not edit `ovms.yaml`.

> **Note on naming.** `fetch` takes the rig-local short name (`my-model`), not the HF repository path. The indirection separates *identity* (HF source + task, declared once under `repository:`) from *references* to that identity (`models[].source`, `graph.draft_model`). Reusing one model in multiple roles, or swapping its upstream source, is a one-line edit instead of a multi-place rename.

## Commands

| Command              | What it does                                                                                                                                               |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `status`             | Reconciles declaration vs world: binary, models, port, live-vs-YAML drift. Read-only.                                                                      |
| `fetch <repo>`       | Pulls a single `repository:` entry via `ovms --pull`. Idempotent. Extra arguments are forwarded to `ovms --pull` verbatim.                                 |
| `activate [profile]` | Activates a profile (or re-applies the current one if no argument is given). Re-renders `config.json`, `graph.<model>.pbtxt` and `generation_config.json`. |
| `deactivate`         | Clears the active flag on all profiles: the live config becomes empty, there is nothing for OVMS to load.                                                  |
| `remove <repo>`      | Removes a model's artifacts (directory + `config.json` entry). Refuses by default if any profile references the model; `--force` ignores references.       |
| `start`              | Runs blocking probes (binary, models, port) and then foreground `exec ovms`. No daemon mode -- wrap it in NSSM/systemd if you need one.                    |

Global flags: `--config`, `--local`, `--ovms-path`, `--log-level`.

## OpenVINO device properties and `CACHE_DIR`

`graph:` has a `plugin_config` field -- a generic dict that is passed through into `LLMCalculatorOptions.plugin_config` and from there to OpenVINO as device properties (`KV_CACHE_PRECISION`, `PERFORMANCE_HINT`, `NUM_STREAMS`, ...). The utility does not validate it -- it is a pass-through bag.

An important quirk: at the time of writing, OVMS 2026.1 **does not** propagate `--cache_dir` into the LLM continuous-batching pipeline (tracked in openvinotoolkit/model_server#4230). For the compile cache to actually be used, `CACHE_DIR` has to go into `plugin_config`. The utility does this automatically from `local.runtime.cache_dir` unless you set `CACHE_DIR` yourself.

## Repository layout

```
ovms-worker/
|-- README.md, DECISIONS.md
|-- pyproject.toml
|-- config/
|   |-- ovms.example.yaml      # declaration example (in git)
|   |-- local.example.yaml     # per-host overrides example (in git)
|-- src/ovms_rig/              # package
|   |-- cli.py                 # entry point and dispatcher
|   |-- command.py             # builds argv for ovms
|   |-- config/                # YAML schema and loader
|   |-- env/                   # environment variable prep
|   |-- probes/                # state checks (status, blocking for start)
|   |-- stages/                # fetch, activate/deactivate, remove, start, status
|-- tests/                     # pytest
```

`config/ovms.yaml`, `config/local.yaml`, and any working artifacts (`.cache/`, `.backup/`) are gitignored.

## Status

The e2e happy path works: on the platform listed below the validated configuration reproduces and reaches the expected 11.7 tok/s. Edge cases (unusual profile combinations, recovery when `activate` is interrupted at rare points, etc.) are not covered -- I use it myself and fix things as they come up.

**Linux:** the code in `signals.py` has a POSIX branch and the schema has no Windows assumptions. It should work, but I have not verified it.

## Context

- **Platform:** Intel Arc 140T iGPU + AI Boost NPU, Windows 11.
- **OVMS:** `2026.1.0.72cc0624` (at the time of validation).
- **Benchmark:** Qwen3-14B-int8 (GPU) + Qwen3-0.6B-int8 (CPU draft, `num_speculative_tokens=7`) -> 11.7 tok/s vs 4.93 tok/s no-speculation.

## License

MIT.
