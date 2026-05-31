# Architectural decisions

A log of the decisions sitting at the foundation of `ovms-rig`. Each one follows a "context / decision / why / alternatives / consequences" structure, with sections dropped when there is nothing to add.

The point of the document is so that future-me (and any reader) understands not only **what** the utility does, but **why it is the way it is** rather than something else. Many of these decisions are not obvious from the code - some of them emerged after bouncing off a string of dead ends.

---

## 1. The shape of the declaration

### 1.1. Mirror the OVMS domain, not my task

**Decision.** The YAML describes OVMS entities (models, endpoints, graphs), not "a speculative-decoding setup for Qwen". The utility brings up any valid OVMS configuration; it does not know about speculative decoding as a separate concept.

**Why.** If the shape is bound to a task, any change to the task (different model, no draft, an embedding service alongside) breaks the abstraction. Mirroring the domain keeps the shape stable: only the property values change.

**Consequence.** There is no `speculative: true` field in the YAML. Speculative decoding is **emergent** - it exists when a `model.graph` has `draft_model` + `draft_device` set. Remove those two fields and you get a plain single-model setup through the same machinery (see `schema.py:Graph._draft_fields_are_paired`).

### 1.2. Three layers: `repository` / `models` / `profiles`

**Decision.** Three levels instead of one flat list of endpoints:

- **repository:** model identities (a weights source -- fetched from an external location or read from a local directory -- plus an optional task). Declared once; the same model can appear in several endpoints.
- **models:** HTTP endpoints. Each one references a `repository` entry via `.source` and carries its own `graph` (`LLMCalculatorOptions` fields).
- **profiles:** named groups of endpoints (a list of names from `models:`). Only one profile can be active.

**Why.** Without the `repository` layer we would have HF identifier duplication everywhere a model is mentioned (as a target and as a draft in another endpoint). Without the `profiles` layer, switching between configurations ("14B speculative" vs "8B flat for debug") would require editing `models:` by hand or juggling git branches.

**Alternative.** A single flat list of endpoints with inline models (the OVMS-native `config.json` shape). Rejected - naming and reuse are lost.

### 1.3. Profiles + exactly one active; a switch, not "apply a new config"

**Decision.** Each profile carries an `active: true | false` flag. Invariant: at most **one** is active at a time. `activate <name>` clears the flag everywhere and sets it on `<name>`; `deactivate` clears it everywhere (the live config becomes empty). Bare `activate` with no argument re-applies the currently active profile (for when you've edited ovms.yaml and want to re-apply the configuration).

**Why.** "One of N" is a frequent operation (switching configurations), and it should not require editing YAML. YAML editing stays for **changing** a profile, not for **picking** between them.

**Consequence.** `ovms-rig` owns the `active:` field in `ovms.yaml` - it writes to it during activate/deactivate. The YAML is no longer purely a "human declaration"; part of its state is maintained by the tool. We accept this: the alternative - a separate state file - adds another sync point and source-of-truth conflicts.

### 1.4. Speculative decoding as an emergent property

See 1.1. Worth stating separately: this is the central idea behind the shape of the declaration. The utility **does not know** what speculative decoding is. It knows that `graph:` contains a bag of fields to patch into `LLMCalculatorOptions`. The semantics are OVMS's concern.

### 1.5. Model source: a fetched-or-local abstraction, not a fixed list

**Decision.** A `repository` entry declares where its weights come from, as exactly one source kind, along one stable axis:

- **fetched** -- pulled from an external location. Today: `hf:` (a HuggingFace id, via `ovms --pull`).
- **local** -- read from a directory the user manages, never fetched: `dir:` (absolute, or relative to `local.models.repository_path`).

The validator enforces exactly-one and treats the kinds as an open set: adding a kind is a one-line change here, while the rest of the docs -- which speak of "a fetched or local source", not of specific kinds -- stay correct.

**Why.** `ovms --pull` only understands HuggingFace, so `hf` is permanently the "pull from HF" coordinate -- it cannot mean anything else. But not every model comes from HF: plain (non-task) models are often dropped into a directory by hand, and earlier the only way to express that was to abuse `hf` as a bare directory name -- a field whose name promised HuggingFace while nothing was pulled. Naming the source by category (fetched/local) rather than by a closed list makes `hf` honest (one option among several), gives local models an explicit field, and keeps both schema and prose extensible.

**Alternative.** A discriminated union (`source: {kind: hf, ref: ...}`). Rejected: heavier and more verbose than the codebase's flat mutually-exclusive-optionals style (already used for `graph`/`plain`), and it would force renaming the unrelated `models[].source` reference field.

**Consequence.** `fetch` is a no-op for any local source. The source kind is independent from `task`: a model of any source kind may be task-based or plain.

**Deferred.** `github` -- rig fetching a release/repo itself, beyond OVMS's HF-only pull -- is simply another *fetched* kind: it joins the list in this section and needs no changes elsewhere.

### 1.6. Plain (non-task) models: a loose `plain:` pass-through

**Decision.** A non-task model (no `task`) is served through OVMS `model_config_list` instead of a mediapipe graph. Its options (`shape`, `layout`, `batch_size`, `nireq`, `model_version_policy`, `allow_cache`, ...) go in a `plain:` block that rig forwards verbatim, without validating keys -- exactly like `plugin_config`.

**Why.** The `model_config_list` option set is large and tracks upstream OVMS; typing it would be a permanent chase. OVMS validates these at load, so rig stays a pass-through. Contrast `graph:`, whose `LLMCalculatorOptions` set is small and stable enough to type. `plain` and `graph` are mutually exclusive: a model is either task/graph or plain.

---

## 2. Managing OVMS files

### 2.1. Sibling copies `graph.<name>.pbtxt`, pristine never mutated

**Decision.** `ovms --pull` generates `graph.pbtxt` in the model directory; we **do not touch it**. For every active endpoint we create a sibling copy `graph.<endpoint_name>.pbtxt` next to it, and the patch goes into the copy. `config.json` points at the copy through `graph_path`.

**Why.** The pristine file is OVMS's contract with its own models. If we mutate it, we get out-of-sync with the pull, trouble on reinstall, an unclear change log. A sibling is reversible: delete the copy, the endpoint is rolled back without a trace.

**Alternative from the draft README.** Edit in place + `.backup/<timestamp>/`. Rejected: a timestamped backup is "we hope we will not need to roll back"; siblings are "rollback is free by construction".

**Consequence.** Multiple endpoints on the same model means multiple `graph.X.pbtxt`, `graph.Y.pbtxt` files side by side, each with its own parameters. `cleanup.py` sweeps siblings whose endpoints have left the active profile.

### 2.2. `config.json` is owned by rig: rewrite, not merge

**Decision.** `config.json` is always rewritten **as an exact projection** of the active profile. Any pre-existing content (entries from other tools, garbage) is discarded.

**Why.** Declarative contract: what is in `ovms.yaml` is what is in `config.json`. Any merge turns the contract into "some-of-mine + some-of-the-neighbor's", and it becomes impossible to tell who owns what.

**Consequence.** We do not coexist peacefully with another tool that writes into the same `config.json`. Documented in `registry.py`.

### 2.3. Textual patcher for `LLMCalculatorOptions`, not `google.protobuf.text_format`

**Decision.** The `graph.pbtxt` editor works with regexes over the text block `[type.googleapis.com/mediapipe.LLMCalculatorOptions]: { ... }` - it replaces existing fields in place and appends missing ones before the closing brace.

**Why.** The canonical path is `text_format.Parse` with a `.proto` schema. The `.proto` for `LLMCalculatorOptions` exists in the OVMS source tree, so the canonical path is technically reachable - but it requires either vendoring the `.proto` (and pinning rig to a specific OVMS version) or pulling it at runtime from a specific upstream commit. Rig deliberately stays version-agnostic: any reasonably recent OVMS should work without requiring a rig re-release. Separately, a `text_format` round trip normalizes field order and formatting, ruining the pristine cosmetics.

**Alternative.** `txtpbfmt` (a schemaless parser). Workable, but adds one more dependency for two regexes.

**Consequence.** The patcher is fragile to changes in OVMS's `pbtxt` format. We catch this via tests: when OVMS upgrades, we run `ovms --pull` on a sample and check that the patcher still hits the right spots.

### 2.4. `.orig` snapshot for `generation_config.json`

**Context.** Unlike `graph.pbtxt` - where `graph_path` in the mediapipe config entry lets several endpoints point at different graph files inside the same model directory - OVMS has no equivalent override for `generation_config.json`. The LLM calculator reads it from a fixed filename inside `models_path`. Filed as upstream feature request [openvinotoolkit/model_server#4233](https://github.com/openvinotoolkit/model_server/issues/4233).

**Decision.** Right after `ovms --pull` we snapshot `generation_config.json` into an `.orig` copy. On `activate` we render the live `generation_config.json` as `merge(.orig, ovms.yaml:models.<name>.generation)`. On `deactivate` we restore the live file from `.orig`.

**Why.** Since there is no per-endpoint override path, applying user `generation:` overrides means rewriting the on-disk file in place. To stay reversible and re-applyable from any state, we need a pristine reference that never moves; `.orig` is exactly that snapshot.

An earlier iteration used a timestamped `.bak` written on every `activate`. That turned the model directory into an archive and made the "right" rollback target ambiguous. `.orig` is a single pristine snapshot per model (taken once by `fetch`), so the merge is always deterministic and the rollback target is unambiguous.

**Consequence.** If the pristine `generation_config.json` is updated (for example, after `ovms --pull --overwrite_models`), `.orig` needs to be re-taken. `fetch` does this automatically.

**Caveat.** Until openvinotoolkit/model_server#4233 lands, two models backed by the same repository entry cannot simultaneously have different generation defaults - the live `generation_config.json` is a single file. Switching between profiles re-renders it, so the active profile's generation settings win.

---

## 3. CLI behavior

### 3.1. `fetch` is per-repository, not batch

**Decision.** `ovms-rig fetch qwen3-14b` pulls exactly one entry from `repository:`. Not `fetch all`, no implicit dependency on a profile. Extra arguments are forwarded to `ovms --pull` verbatim.

**Why.** Pulling a large model is a slow and fragile operation (network, disk, interruptions). Making it part of a composite command masks which exact pull failed and why. Explicit per-repo control is better than all-or-nothing.

**Consequence.** Quick start includes explicit `fetch` calls for each needed repo. That is fine: explicitness has a cost.

### 3.2. `remove --force` to override profile references

**Decision.** `ovms-rig remove <repo>` deletes the model directory on disk and the corresponding entry from `config.json`. It does **not** touch `ovms.yaml` (the `repository`, `models`, or `profiles` sections stay as the user wrote them).

By default, before deleting anything, `remove` scans **all profiles** (active or not) for references to the repository: a profile's `models[].source` matches, or another endpoint's `graph.draft_model` matches. If any reference is found, `remove` refuses and lists the offending profiles. `--force` skips the scan.

**Why.** Removing a model that *any* profile references is almost always a mistake. The active profile being broken is the obvious case (the next `start` will not bring the server up), but an inactive profile that becomes unbootable on activation is just as broken; it only fails later. Better to fail at `remove` than at the next `activate`.

`--force` exists because sometimes the workflow is "remove first, then clean the YAML" - intentionally in that order, e.g. when freeing disk before editing the declaration.

### 3.3. `start` - foreground exec, no daemon mode

**Decision.** `start` blocks on the ovms process and forwards SIGTERM/SIGINT (POSIX) or CTRL_BREAK/terminate (win32). No `--detach`, no `--daemon`, no pid-files.

**Why.** Daemon logic is its own complexity (where to keep the pid, what to do on crash, how to read the stdout of an already-running instance). All of these are solved better by external tools (NSSM on Windows, systemd on Linux). We lean on those.

**Consequence.** To run under NSSM/systemd, point them at `ovms-rig start` as the entry point. Signals arrive correctly, graceful shutdown works (see `signals.py`).

### 3.4. Managed-flags blocklist on `start` extras

**Decision.** Some ovms flags (`--log_level`, `--log_path`, `--config_path`) are forbidden in `start` extras. Passing one of them raises `ValueError`.

**Why.** Rig uses these flags itself: `--config_path` points at the generated `config.json`, `--log_level` comes from the YAML or the global rig CLI, `--log_path` is governed by the choice to keep logs on stdout (not a file). Allowing them in extras would make rig's behavior dependent on what the user appends.

### 3.5. ovms binary resolver

**Decision.** Search order: `--ovms-path` (rig CLI flag) -> `runtime.ovms_path` from `local.yaml` -> `which ovms` / `Get-Command ovms` on `PATH` -> error with instructions.

**Why.** No implicit `$OVMS_HOME` or similar conventions - there are too many OVMS install patterns, and any implicit default only makes things worse. Explicit priority, clear diagnostics.

---

## 4. Technology choices

### 4.1. Python + uv + click + pydantic, not Go or Rust

**Decision.** Implementation in Python 3.12+, via `uv` (lock + tool install), CLI on `click`, schema on `pydantic`.

**Why.** This project's ecosystem is Python. `huggingface_hub` is a Python SDK; the `pbtxt` editor, even though we went with regex, is easy to extend to `google.protobuf.text_format` if needed. In Go/Rust we would have to drag in reinvented wrappers.

**Alternatives:**

- **C++.** Same language as OVMS, but rig is an external orchestrator: it shells out and patches text files, never links against OVMS. Shared language pays off with tight integration (embed, shared types), not over a CLI-and-files surface. The one real win - a proto-based pbtxt patcher - was rejected anyway (see 2.3).
- **Rust.** Over-engineering for a thin wrapper. Here, Rust's type system would not carry stronger invariants than the pydantic schema already does.
- **Go.** Fine, but foreign to the ML ecosystem: we would have to fight it instead of using it.

### 4.2. Strict pydantic schema (`extra="forbid"`)

**Decision.** Any typo in YAML - fail at load, not mid-pipeline.

**Why.** Silently ignoring a typo means hours spent figuring out why `enable_prefix_caching` does not work (it is spelled `enabled_prefix_caching` in YAML). Better to fail loudly at startup.

### 4.3. Applications vs. artifacts

**Decision.** The OVMS binary is an **application**. The utility does not install it - it only locates it and writes a clear error if it cannot find one. Models are **artifacts**. The utility pulls them on a clean machine.

**Why.** Applications are usually installed by the OS package manager or manually from a release archive; doing that ourselves would just be reinventing the wheel. Artifacts, on the other hand, are typically pulled by the application itself or its wrapper.

---

## 5. Reliability

### 5.1. `activate` failure modes: transactional apply, non-transactional smoke-load

**Decision.** `activate` has two failure paths, treated differently on purpose.

**Path A: apply-internal failure (transactional).** Before the apply substage starts writing, it takes an in-memory snapshot of the current `config.json`. If any error occurs mid-write (I/O failure, malformed graph template, bug in the patcher), apply rolls back: `config.json` is restored from the snapshot and any sibling graphs created in this run are deleted. The disk returns to its pre-`activate` state. The snapshot was taken from a known-coherent on-disk state, and the rollback target is unambiguous.

**Path B: smoke-load failure (non-transactional).** After all files are written, a smoke-load probe launches ovms briefly on the resulting configuration. If ovms refuses to load (version mismatch, incompatible field, etc.), rig logs the error and exits non-zero, but **does not** touch the rendered files. They stay on disk as written, broken, until the user fixes the YAML and re-runs `activate`.

**Why path B is not transactional.** An earlier version did try to roll back here: snapshot `ovms.yaml` before activate -> on smoke failure, restore `ovms.yaml` + re-render derived from the snapshot. The idea was "return to the last known-good state". But under the real workflow, where `ovms.yaml` is edited by hand and `activate` is the validation step, the snapshot does not capture a known-good state, it captures the unvalidated user edit. "Rolling back" to it just re-applied the broken input, masked the error with a "rollback completed" log line, and forced the user to debug a stale derived state on top. The outcome was identical to doing nothing, plus misleading logs.

So path B is now explicit: smoke-load said no, derived files are stale, fix the YAML and re-run.

**Consequence.** "Atomic activate" is a slogan, not the contract. The actual contract is "apply substage is transactional; smoke-load is a probe, not a gate". A red smoke-load means the next `start` will fail in the same way the smoke probe failed - the user is told upfront instead of finding out later.

Smoke-load briefly launches a real ovms process, which is much heavier than the in-process probes. We pay that cost on `activate` (a state change anyway), but not on `status` (read-only).

### 5.2. Probes registry with presets

**Decision.** All state checks (`bin`, `models`, `port`, `live_config`, `model_files`, ...) live in `probes/registry.py`, each as a named function. `status` runs the `DIAGNOSTIC` preset (everything); `start` runs the `BLOCKING` preset (only what makes startup impossible if missing).

**Why.** Duplication between "what we check in status" and "what we check before start" led to drift: something added in one place was forgotten in the other. A single registry plus presets removes that.

---

## 6. Log of open questions

The first draft of the README carried a list of questions I did not have answers to at the start. Here is what happened to each of them.

### Device assignment in `ovms --pull`

**Question.** There is no `--target_device` flag in pull mode; `device` and `draft_device` live inside `graph.pbtxt`. What does pull set by default?

**Answer.** Pull sets some default (usually `CPU`); rig overwrites `device` and `draft_device` from the declaration during `activate` anyway. The subtlety of pull's behavior turned out to be irrelevant.

### Does OVMS publish a `.proto` for `LLMCalculatorOptions`

**Question.** This determined the approach to the `pbtxt` editor.

**Answer.** Yes - in the OVMS source tree, but vendoring it would couple rig to a specific OVMS version. We chose the textual patcher (see 2.3).

### Why `--cache_dir` is not populated

**Question.** Cache files did not appear on the first start.

**Answer.** OVMS 2026.1 does not propagate `--cache_dir` into the LLM continuous-batching pipeline - the global flag is logged as enabled but never reaches the underlying `ContinuousBatchingPipeline`. Filed upstream as [openvinotoolkit/model_server#4230](https://github.com/openvinotoolkit/model_server/issues/4230). Workaround: put `CACHE_DIR` into `plugin_config`. The utility does this automatically (see the README, "OpenVINO device properties and CACHE_DIR").

### Behavior of `ovms --pull` on an already-downloaded model

**Answer.** Without `--overwrite_models` pull is idempotent (checks presence, does not re-download). `fetch` in rig relies on this.
