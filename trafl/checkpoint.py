"""JSON artifacts, атомарные full checkpoints и восстановление RNG streams."""

import json
import os
import random

import yaml


def write_json(path, value):
    r"""Сохранить JSON-артефакт UTF-8-readable представлением.
    
    Args:
        path: pathlib.Path целевого файла; родительский каталог уже создан caller.
        value: JSON-serializable данные (metrics, IDs, versions или answers).
    
    Returns:
        None. Существующий файл заменяется через Path.write_text; текст сохраняет
        Unicode, использует indent=2 и завершается newline.
    
    Atomic full-training checkpoint записывается отдельно в save_checkpoint."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")



def restore_rng(state, rng, mask_rng):
    r"""Восстановить все main-process и rollout/masking RNG streams.
    
    Args:
        state: Checkpoint rng dict: python, numpy, torch_cpu, torch_cuda,
            rollout и mask; tensor RNG states загружаются на CPU.
        rng: Уже созданный device generator diffusion rollouts.
        mask_rng: Уже созданный CPU generator training corruptions.
    
    Returns:
        None. Глобальные RNG и generators изменяются in-place.
    
    Runner вызывает это после model/head initialization, data preparation,
    optimizer restore и worker construction, чтобы bootstrap не расходовал
    сохранённые draws. Shuffle/worker RNG восстанавливаются через loader state."""
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("cuda_local"):
        if state["torch_cuda"]:
            torch.cuda.set_rng_state(state["torch_cuda"][0])
    else:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    rng.set_state(state["rollout"])
    mask_rng.set_state(state["mask"])



def save_checkpoint(
    policy,
    head,
    cfg,
    out,
    step,
    optimizer,
    rng,
    mask_rng,
    loader_state,
    running,
    accumulated,
    record,
):
    r"""Сохранить атомарный full resume snapshot и evaluation export.
    
    Args:
        policy, head: Current LoRA policy и log Z head.
        cfg, out, step: Effective config, run directory и completed update count.
        optimizer: Joint optimizer, включая moments и param-group state.
        rng, mask_rng: Текущие device rollout / CPU mask generators.
        loader_state: Explicit epoch permutation, processed-batch cursor и RNG.
        running, accumulated: Накопленные scalars и число updates до logging flush.
        record: Train JSONL stream; flush/fsync и position входят в snapshot.
    
    Returns:
        None. checkpoint/training_state.pt содержит policy/head/optimizer,
        original scheduler horizon/next update, selected IDs, RNG и logging context.
        Parallel checkpoint дополнительно содержит rank_states: RNG, loader,
        record cursor и running metrics каждого rank. Только rank zero пишет
        модель/optimizer; остальные ждут атомарной публикации.
    
    После torch.save в temporary file выполняются fsync, atomic os.replace
    и fsync каталога: прерванная запись сохраняет previous authoritative snapshot.
    Затем отдельно экспортируются adapters, head.pt и config.yaml для evaluation.
    Resume/new evaluation читают training_state.pt как согласованное состояние.
    Сохранять нужно между полными optimizer updates, не внутри rollout backward."""
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import get_peft_model_state_dict

    checkpoint = out / "checkpoint"
    record.flush()
    os.fsync(record.fileno())
    local = {
        "loader": loader_state,
        "running": running.detach().cpu(),
        "accumulated": accumulated,
        "record_position": record.tell(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [torch.cuda.get_rng_state()] if torch.cuda.is_available() else [],
            "cuda_local": True,
            "rollout": rng.get_state(),
            "mask": mask_rng.get_state(),
        },
    }
    parallel = dist.is_initialized()
    rank = dist.get_rank() if parallel else 0
    rank_states = [None] * dist.get_world_size() if parallel and rank == 0 else None
    if parallel:
        # Only small runtime state crosses ranks; model/optimizer are identical.
        dist.gather_object(local, rank_states, dst=0)
        if rank != 0:
            dist.barrier()
            return
    checkpoint.mkdir(exist_ok=True)
    state = {
        "policy": get_peft_model_state_dict(policy),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": dict(cfg),
        "scheduler": {"next_update": step, "horizon": cfg["updates"]},
        "selected_examples": json.loads((out / "selected_examples.json").read_text()),
        **local,
    }
    if parallel:
        state["rank_states"] = rank_states
    # Self-contained authoritative resume state; old state survives interrupted writes.
    temporary = checkpoint / "training_state.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint / "training_state.pt")
    directory = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(directory)
    os.close(directory)
    # Also keep the familiar standalone evaluation export.
    policy.save_pretrained(checkpoint)
    torch.save({"head": head.state_dict(), "step": step}, checkpoint / "head.pt")
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    if parallel:
        dist.barrier()

