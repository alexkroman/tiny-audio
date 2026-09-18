"""Estimate the GPU memory and disk a training config will actually need.

Provisioning is the one step where guessing is expensive: too small a GPU and
the run OOMs an hour into dataset prep, too small a container disk and the
weights download dies at 95%. This reads the resolved Hydra config and reports
both, then emits a ready-to-run `runpodctl pod create`.

Everything that can be measured is measured rather than assumed:
  - parameter counts come from safetensors headers over HTTP range requests
    (no weights downloaded),
  - download sizes come from the Hub's file metadata,
  - the projector is instantiated from the real PROJECTOR_CLASSES entry so the
    number can't drift from the implementation.

Activation memory is the one genuine estimate; its formula is printed so the
number can be argued with rather than trusted blindly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import typer

GIB = 1024**3
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
# AdamW keeps exp_avg + exp_avg_sq per trainable parameter.
OPTIMIZER_STATES = 2
# Headroom for fragmentation, cuBLAS workspaces, NCCL buffers and the CUDA
# context. 1.25 is deliberately modest; raise it if runs OOM near the estimate.
OVERHEAD_FACTOR = 1.25
# Rough size of the installed python env + apt packages on the pod.
ENV_DISK_GIB = 12.0
# datasets keeps two persistent copies of every source, and neither is cleaned
# up during the run: the Hub download lands as parquet in the hub cache under
# HF_HOME, and load_dataset() then writes its own Arrow tables under
# data.dataset_cache_dir. Measured on hf-internal-testing/librispeech_asr_dummy
# (datasets 4.x): 8.99 MB of parquet in the hub cache plus 9.46 MB of Arrow in
# the cache_dir, i.e. 2.05x the download size. Charging the download once is
# how a correctly-sized pod still dies with ENOSPC halfway through prep.
DATASET_DISK_FACTOR = 2.05


@dataclass
class Component:
    name: str
    params: int = 0
    download_bytes: int = 0
    trainable: bool = False
    note: str = ""


@dataclass
class Plan:
    components: list[Component] = field(default_factory=list)
    dataset_bytes: int = 0
    dataset_rows: list[tuple[str, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vram: dict[str, float] = field(default_factory=dict)
    disk: dict[str, float] = field(default_factory=dict)


def _safetensors_params(repo_id: str) -> tuple[int, str]:
    """Exact parameter count from the safetensors header (no weight download)."""
    from huggingface_hub import get_safetensors_metadata

    meta = get_safetensors_metadata(repo_id)
    counts = meta.parameter_count
    # Ignore integer buffers (rotary caches, position ids); they aren't params.
    total = sum(n for dtype, n in counts.items() if not dtype.startswith("I"))
    dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "?"
    return total, dominant


def _vocab_table_params(text_cfg) -> dict[str, int]:
    """Per-token lookup-table sizes for a decoder, computed from its config.

    These can be frozen independently of the rest of the decoder
    (`freeze_text_embed_tokens`), and on Gemma 4 they are the majority of the
    checkpoint -- so counting them as trainable overstates AdamW state badly.
    Empty entries are omitted, so decoders without a per-layer table simply
    don't report one.
    """
    tables: dict[str, int] = {}
    vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
    hidden = int(getattr(text_cfg, "hidden_size", 0) or 0)
    if vocab and hidden:
        tables["embed_tokens"] = vocab * hidden
    ple_vocab = int(getattr(text_cfg, "vocab_size_per_layer_input", 0) or 0)
    ple_hidden = int(getattr(text_cfg, "hidden_size_per_layer_input", 0) or 0)
    layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    if ple_vocab and ple_hidden and layers:
        tables["embed_tokens_per_layer"] = ple_vocab * layers * ple_hidden
    return tables


def _repo_weight_bytes(repo_id: str, repo_type: str = "model", name: str | None = None) -> int:
    """Total bytes the Hub will hand us for a repo, optionally one config only.

    Multi-config dataset repos (libriheavy ships small/medium/large) would be
    wildly overcounted by summing every shard, so when a config `name` is given
    we keep only files whose path mentions it. Falls back to the full repo when
    nothing matches, since a silent zero would be worse than an overestimate.
    """
    from huggingface_hub import dataset_info, model_info

    info = (model_info if repo_type == "model" else dataset_info)(repo_id, files_metadata=True)
    siblings = [s for s in info.siblings if (s.size or 0) > 0]
    if name:
        scoped = [s for s in siblings if name.lower() in s.rfilename.lower()]
        if scoped:
            siblings = scoped
    return sum(s.size or 0 for s in siblings)


def _load_cfg(experiment: str, overrides: list[str]):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    configs = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        return compose(config_name="config", overrides=[f"+experiments={experiment}", *overrides])


def _hidden(cfg_obj) -> int | None:
    for attr in ("hidden_size", "d_model"):
        if getattr(cfg_obj, attr, None):
            return getattr(cfg_obj, attr)
    return None


def build_plan(experiment: str, overrides: list[str], seq_len: int) -> Plan:
    from transformers import AutoConfig

    from tiny_audio.asr_config import ASRConfig
    from tiny_audio.projectors import PROJECTOR_CLASSES

    cfg = _load_cfg(experiment, overrides)
    plan = Plan()

    model_dtype = str(cfg.model.get("model_dtype", "bfloat16"))
    bytes_per = DTYPE_BYTES.get(model_dtype, 2)
    train = cfg.training

    audio_id = str(cfg.model.audio_model_id)
    text_id = str(cfg.model.text_model_id)

    # ---- encoder -----------------------------------------------------------
    enc_params, enc_dtype = _safetensors_params(audio_id)
    enc_trainable = not train.get("freeze_audio_encoder", True)
    plan.components.append(
        Component(
            f"encoder ({audio_id})",
            enc_params,
            _repo_weight_bytes(audio_id),
            enc_trainable,
            f"checkpoint {enc_dtype}",
        )
    )

    # ---- decoder -----------------------------------------------------------
    dec_params, dec_dtype = _safetensors_params(text_id)
    dec_trainable = not train.get("freeze_language_model", True)
    dec_cfg = AutoConfig.from_pretrained(text_id)
    text_cfg = dec_cfg.get_text_config() if hasattr(dec_cfg, "get_text_config") else dec_cfg
    # Split the frozen vocabulary table out of the trainable decoder. The
    # freeze flag acts on an individual tensor inside the language model, so a
    # single all-or-nothing `trainable` on one component would charge AdamW
    # state for parameters that never see the optimizer on this recipe.
    frozen_tables: dict[str, int] = {}
    if dec_trainable:
        tables = _vocab_table_params(text_cfg)
        if train.get("freeze_text_embed_tokens", False) and tables.get("embed_tokens"):
            frozen_tables["embed_tokens"] = tables["embed_tokens"]
    frozen_table_params = sum(frozen_tables.values())

    plan.components.append(
        Component(
            f"decoder ({text_id})",
            dec_params - frozen_table_params,
            _repo_weight_bytes(text_id),
            dec_trainable,
            f"checkpoint {dec_dtype}",
        )
    )
    if frozen_table_params:
        plan.components.append(
            Component(
                f"  frozen tables ({', '.join(frozen_tables)})",
                frozen_table_params,
                0,  # already charged by the decoder's repo download above
                False,
                "no optimizer state",
            )
        )

    # ---- projector ---------------------------------------------------------
    enc_cfg = AutoConfig.from_pretrained(audio_id)
    enc_inner = getattr(enc_cfg, "encoder_config", None) or enc_cfg
    encoder_dim = _hidden(enc_inner)
    llm_dim = _hidden(text_cfg)
    proj_params = 0
    if encoder_dim and llm_dim:
        shim = ASRConfig(
            audio_model_id=audio_id,
            text_model_id=text_id,
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            projector_type=str(cfg.model.get("projector_type", "mlp")),
            projector_pool_stride=int(cfg.model.get("projector_pool_stride", 4)),
            projector_hidden_dim=cfg.model.get("projector_hidden_dim"),
        )
        cls = PROJECTOR_CLASSES[shim.projector_type]
        proj_params = sum(p.numel() for p in cls(shim).parameters())
    else:
        plan.warnings.append("Could not resolve encoder/llm dims; projector excluded.")
    plan.components.append(
        Component("projector (fresh init)", proj_params, 0, True, f"{encoder_dim}->{llm_dim}")
    )

    # ---- datasets ----------------------------------------------------------
    for entry in cfg.data.get("datasets", []) or []:
        path = str(entry.get("path"))
        name = entry.get("name")
        if not entry.get("train_splits") and not entry.get("eval_splits"):
            continue
        try:
            size = _repo_weight_bytes(path, "dataset", str(name) if name else None)
        except Exception as exc:  # gated repos, renames, network
            plan.warnings.append(f"dataset {path}: size unknown ({type(exc).__name__})")
            continue
        plan.dataset_bytes += size
        plan.dataset_rows.append((f"{path}" + (f":{name}" if name else ""), size))

    # ---- VRAM --------------------------------------------------------------
    total_params = sum(c.params for c in plan.components)
    trainable_params = sum(c.params for c in plan.components if c.trainable)

    # Trainable params may be held at a different (higher) precision than the
    # frozen stack; see ASRConfig.projector_dtype. Charging everything at the
    # trainable dtype overstated weights by 10.4 GiB on this recipe.
    proj_dtype = str(cfg.model.get("projector_dtype") or model_dtype)
    trainable_bytes_per = DTYPE_BYTES.get(proj_dtype, bytes_per)
    frozen_params = total_params - trainable_params
    weights = frozen_params * bytes_per + trainable_params * trainable_bytes_per
    grads = trainable_params * trainable_bytes_per
    optim = trainable_params * trainable_bytes_per * OPTIMIZER_STATES

    batch = int(train.get("per_device_train_batch_size", 1))
    layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    inter = int(getattr(text_cfg, "intermediate_size", 0) or 0)
    vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
    ckpt = bool(train.get("gradient_checkpointing", False))

    # Per token per layer: attention q/k/v/o + residual (~6*hidden) and the
    # MLP's gate/up/down (~3*intermediate). Coarse but the right order.
    per_tok_layer = bytes_per * (6 * (llm_dim or 0) + 3 * inter)
    if ckpt:
        # Only layer boundaries are kept; one layer is recomputed at a time.
        acts = (
            batch * seq_len * (llm_dim or 0) * layers * bytes_per + batch * seq_len * per_tok_layer
        )
    else:
        acts = batch * seq_len * layers * per_tok_layer

    # Cross-entropy. liger fuses lm_head+softmax+CE into O(B*T*D); without it
    # the (B, T, V) fp32 logits plus a log_softmax copy dominate everything.
    fused = bool(train.get("use_liger", True))
    logits = 0 if fused else batch * seq_len * vocab * 4 * 2

    subtotal = weights + grads + optim + acts + logits
    plan.vram = {
        "weights": weights / GIB,
        "gradients": grads / GIB,
        "optimizer (AdamW x2)": optim / GIB,
        "activations (est.)": acts / GIB,
        "cross-entropy": logits / GIB,
        "subtotal": subtotal / GIB,
        "recommended (x1.25)": subtotal * OVERHEAD_FACTOR / GIB,
    }

    # Gradients must reach the projector, which sits at the *input* of the
    # decoder, so every decoder layer's activations are retained even though
    # the decoder itself is frozen. Freezing saves optimizer state, not
    # activation memory -- a common and expensive surprise.
    # Two known over-counts, both harmless while the decoder was frozen (the
    # projector was the only trainable module) and both material once it is not.
    # Stated rather than corrected: erring high is the safe direction for pod
    # sizing, but a silently inflated figure invites renting the wrong GPU.
    if dec_trainable:
        plan.warnings.append(
            "Trainable-parameter count is an upper bound: it includes any "
            "multimodal tower in the checkpoint that ASRModel discards "
            "(~476M on gemma-4-E2B-it), and charges trainable weights and "
            "gradients at projector_dtype even though the decoder trains at "
            "model_dtype. Expect real usage below the figure above."
        )

    if not dec_trainable and trainable_params and not ckpt:
        plan.warnings.append(
            "Decoder is frozen but the projector feeds its input, so activations "
            "are still held for every decoder layer. gradient_checkpointing=true "
            "is the lever if activations dominate."
        )
    if not fused:
        plan.warnings.append(
            f"use_liger is off: unfused CE over vocab={vocab:,} adds "
            f"{logits / GIB:.1f} GiB at batch={batch}, seq={seq_len}."
        )

    # A checkpoint is model.safetensors + optimizer.pt, and both scale with the
    # whole *trainable* stack rather than the projector: ASRModel.state_dict
    # serializes the language model too whenever freeze_language_model is false
    # (stage_1), and HF Trainer always writes AdamW's two states per trainable
    # param. save_total_limit copies sit on disk simultaneously, so retention
    # multiplies -- charging one projector-sized checkpoint understated a joint
    # fine-tune by ~500x.
    keep = max(int(train.get("save_total_limit", 1) or 1), 1)
    ckpt_each = trainable_params * trainable_bytes_per * (1 + OPTIMIZER_STATES)
    ckpt_bytes = ckpt_each * keep

    weights_bytes = sum(c.download_bytes for c in plan.components)
    datasets_bytes = plan.dataset_bytes * DATASET_DISK_FACTOR
    total = weights_bytes + datasets_bytes + ckpt_bytes
    plan.disk = {
        "model weights": weights_bytes / GIB,
        f"datasets (parquet+arrow x{DATASET_DISK_FACTOR})": datasets_bytes / GIB,
        "python env + apt": ENV_DISK_GIB,
        f"checkpoints ({keep} kept x {_fmt(ckpt_each / GIB)})": ckpt_bytes / GIB,
        "recommended": total / GIB + ENV_DISK_GIB,
    }

    # RunPod container disks are provisioned from the host's local storage and
    # large requests are a common cause of "no instances available"; past ~1 TiB
    # a network volume is the realistic way to satisfy /workspace.
    if plan.disk["recommended"] > 1024:
        plan.warnings.append(
            f"{plan.disk['recommended'] / 1024:.1f} TiB of /workspace is a lot to ask of a "
            "container disk. Attach a network volume (--network-volume-id) or cut the "
            "dataset mix; `datasets` needs room for parquet AND arrow at once."
        )
    return plan


def _fmt(n: float) -> str:
    return f"{n:,.2f} GiB"


def plan_command(
    experiment: str = typer.Option("granite_gemma", "--experiment", "-e"),
    seq_len: int = typer.Option(512, "--seq-len", help="Assumed tokens per sample"),
    gpu: str = typer.Option(
        "NVIDIA H100 80GB HBM3", "--gpu", help="GPU id for the emitted command"
    ),
    image: str = typer.Option("runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "--image"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    overrides: list[str] = typer.Argument(None, help="Extra Hydra overrides"),
):
    """Estimate GPU memory and disk for a training config, and emit a pod command."""
    plan = build_plan(experiment, list(overrides or []), seq_len)

    if as_json:
        print(
            json.dumps(
                {
                    "experiment": experiment,
                    "vram_gib": plan.vram,
                    "disk_gib": plan.disk,
                    "components": [
                        {
                            "name": c.name,
                            "params": c.params,
                            "download_gib": c.download_bytes / GIB,
                            "trainable": c.trainable,
                        }
                        for c in plan.components
                    ],
                    "warnings": plan.warnings,
                },
                indent=2,
            )
        )
        return None

    print(f"\n=== Resource plan: +experiments={experiment} (seq_len={seq_len}) ===\n")
    print(f"{'component':<52} {'params':>14} {'download':>11}  train")
    for c in plan.components:
        print(
            f"{c.name[:52]:<52} {c.params:>14,} "
            f"{c.download_bytes / GIB:>10.2f}G  {'yes' if c.trainable else 'no'}"
        )

    if plan.dataset_rows:
        print(f"\n{'dataset':<52} {'download':>11}")
        for name, size in sorted(plan.dataset_rows, key=lambda r: -r[1]):
            print(f"{name[:52]:<52} {size / GIB:>10.2f}G")

    print("\n--- GPU memory ---")
    for k, v in plan.vram.items():
        print(f"  {k:<38} {_fmt(v)}")
    print("\n--- Disk ---")
    for k, v in plan.disk.items():
        print(f"  {k:<38} {_fmt(v)}")

    for w in plan.warnings:
        print(f"\n  ! {w}")

    disk_gb = int(plan.disk["recommended"] * 1.15) + 5
    print("\n--- Provision ---")
    print(
        f"  runpodctl pod create --name tiny-audio-{experiment} \\\n"
        f'    --gpu-id "{gpu}" --image {image} \\\n'
        f"    --container-disk-in-gb {disk_gb} --ports '22/tcp' \\\n"
        f'    --env "{{\\"SSH_PUBLIC_KEY\\":\\"$(cat ~/.ssh/id_ed25519.pub)\\"}}"\n'
    )
    print(f"  Needs a GPU with >= {plan.vram['recommended (x1.25)']:.0f} GiB VRAM.")
    print(
        "  Disk note: the remote training script exports HF_HOME=/workspace/.cache\n"
        "  and HF_DATASETS_CACHE=/workspace/datasets, so the figure above must be\n"
        "  satisfied by whatever backs /workspace -- the network volume when one is\n"
        "  attached (--network-volume-id), otherwise the container disk. Sizing the\n"
        "  container disk while downloads land on a smaller volume, or vice versa,\n"
        "  is the usual way this fails at 95% of a weights pull.\n"
    )
    return 0


def _available_gpus(min_vram_gib: float) -> list[tuple[int, str]]:
    """GPUs the catalog claims are available with enough VRAM, smallest first.

    Smallest-first approximates cheapest-first; `runpodctl gpu list` exposes no
    price field. The returned order is a candidate list rather than a choice,
    because `available` is not a promise -- see `provision`.
    """
    import subprocess

    out = subprocess.run(
        ["runpodctl", "gpu", "list", "-o", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    catalog = json.loads(out[out.index("[") :])
    fitting = [
        (g["memoryInGb"], g["gpuId"])
        for g in catalog
        if g.get("available") and g.get("memoryInGb", 0) >= min_vram_gib
    ]
    return sorted(set(fitting))


def provision_command(
    experiment: str = typer.Option("granite_gemma", "--experiment", "-e"),
    seq_len: int = typer.Option(512, "--seq-len"),
    name: str | None = typer.Option(
        None, "--name", help="Pod name (default tiny-audio-<experiment>)"
    ),
    image: str = typer.Option("runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "--image"),
    max_attempts: int = typer.Option(6, "--max-attempts", help="How many GPU types to try"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan and candidates, create nothing"
    ),
    overrides: list[str] = typer.Argument(None),
):
    """Size a config, then create a pod on the first GPU type that has capacity.

    RunPod's catalog reports `available: true` and `stockStatus: Low` for GPU
    types that still fail to create with "There are no longer any instances
    available with the requested specifications" -- availability is per
    datacenter and racy, so a single hardcoded --gpu-id fails intermittently.
    This walks the fitting GPU types smallest-first until one actually comes up.
    """
    import subprocess
    from pathlib import Path

    plan = build_plan(experiment, list(overrides or []), seq_len)
    vram = plan.vram["recommended (x1.25)"]
    disk = int(plan.disk["recommended"] * 1.15) + 5
    candidates = _available_gpus(vram)

    print(f"\n{experiment}: needs >= {vram:.1f} GiB VRAM, {disk} GB disk")
    # Surface the same warnings `plan` prints -- the network-volume one in
    # particular explains a container-disk request that RunPod will refuse to
    # fill, which otherwise reads as plain "no capacity" on every GPU type.
    for w in plan.warnings:
        print(f"  ! {w}")
    if not candidates:
        print("No listed GPU type has enough VRAM. Reduce batch size or enable")
        print("gradient_checkpointing, then re-run.")
        raise typer.Exit(1)
    print(f"candidates (smallest first): {', '.join(g for _, g in candidates[:max_attempts])}\n")

    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser()
    if not pubkey.exists():
        print(f"Missing {pubkey}; `ta runpod deploy` authenticates with that key.")
        raise typer.Exit(1)

    if dry_run:
        return None

    pod_name = name or f"tiny-audio-{experiment}"
    for vram_gib, gpu_id in candidates[:max_attempts]:
        print(f"trying {gpu_id} ({vram_gib} GB)... ", end="", flush=True)
        result = subprocess.run(
            [
                "runpodctl",
                "pod",
                "create",
                "--name",
                pod_name,
                "--gpu-id",
                gpu_id,
                "--image",
                image,
                "--container-disk-in-gb",
                str(disk),
                "--ports",
                "22/tcp",
                "--env",
                json.dumps({"SSH_PUBLIC_KEY": pubkey.read_text().strip()}),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        blob = result.stdout + result.stderr
        if "no longer any instances" in blob or '"error"' in blob:
            print("no capacity")
            continue
        try:
            pod = json.loads(blob[blob.index("{") :])
            pod_id = pod["id"]
        except Exception:
            print(f"unexpected response:\n{blob[:400]}")
            continue
        print(f"created {pod_id}")
        print("\nNext:")
        print(f"  poetry run ta runpod wait {pod_id}          # prints <ip> <port>")
        print("  poetry run ta runpod deploy <ip> <port>")
        print(f"  poetry run ta runpod train <ip> <port> -e {experiment} --no-attach -s run1 -f")
        print(f"  runpodctl pod delete {pod_id}               # when finished\n")
        return pod_id

    print("\nEvery candidate GPU type was out of capacity. Retry shortly.")
    raise typer.Exit(1)


def wait_command(
    pod_id: str = typer.Argument(..., help="Pod id from `ta runpod up`"),
    timeout_s: int = typer.Option(900, "--timeout", help="Give up after this long"),
):
    """Block until a pod exposes SSH, then print `<ip> <port>`.

    The endpoint lives at top-level `ssh.ip` / `ssh.port` in the pod JSON.
    `runtime` stays null the whole time on these images, so watching it makes a
    perfectly healthy pod look hung for the 5-10 minutes the image pull takes.
    """
    import subprocess
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = subprocess.run(
            ["runpodctl", "pod", "get", pod_id, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        try:
            pod = json.loads(out[out.index("{") :])
            ssh = pod.get("ssh") or {}
            if ssh.get("ip") and ssh.get("port"):
                print(f"{ssh['ip']} {ssh['port']}")
                return
        except Exception:
            pass
        time.sleep(15)
    print(f"Pod {pod_id} exposed no SSH endpoint within {timeout_s}s.")
    raise typer.Exit(1)
