"""Dataset subsets, Qwen chat tokenization и resume-aware deterministic batch order."""

from .checkpoint import write_json


def collate(rows):
    r"""Сохранить prompt rows списком без padding или stacking.
    
    Args:
        rows: Список предварительно токенизированных словарей DataLoader batch.
    
    Returns:
        Тот же список. Prompt lengths могут отличаться; training/evaluation
        обрабатывают rows последовательно. Collate не потребляет RNG.
    
    Функция остаётся top-level для pickling multiprocessing DataLoader workers."""
    return rows



def prepared_data(cfg, tokenizer, out, selected_ids=None, write_output=True):
    r"""Загрузить и один раз токенизировать train/evaluation prompt subsets.
    
    Args:
        cfg: Mutable flat config. dataset_revision=null разрешается через HfApi;
            resolved revision записывается обратно в cfg.
        tokenizer: Qwen tokenizer с apply_chat_template и enable_thinking.
        out: Run directory для selected_examples.json.
        write_output: Только rank zero пишет shared selected_examples.json.
        selected_ids: Сохранённые split -> example IDs при resume; иначе
            seeded randperm и train_limit/eval_limit выбирают initial subsets.
    
    Returns:
        Пара списков (train, validation). Row содержит CPU input_ids, question,
        answer и original example_id. Prompt template заменяет только {question},
        оставляя literal boxed braces; chat содержит одно user message.
    
    Dataset revision и выбранные IDs обеспечивают воспроизводимую подготовку.
    DataLoader workers позже получают готовые rows без stochastic transforms."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = cfg["dataset_revision"] or HfApi().dataset_info(cfg["dataset_id"]).sha
    cfg["dataset_revision"] = revision
    dataset = load_dataset(cfg["dataset_id"], cfg["dataset_config"], revision=revision)
    result, selected = [], {}
    for split_key, limit_key in (
        ("train_split", "train_limit"),
        ("eval_split", "eval_limit"),
    ):
        split = dataset[cfg[split_key]]
        if selected_ids is None:
            indices = torch.randperm(
                len(split), generator=torch.Generator().manual_seed(cfg["seed"])
            )
            if cfg[limit_key] is not None:
                indices = indices[: cfg[limit_key]]
        else:
            indices = torch.tensor(selected_ids[cfg[split_key]])
        rows = []
        for index in indices.tolist():
            example = split[index]
            text = cfg["prompt_template"].replace("{question}", example["question"])
            # Replace only the task placeholder; literal boxed braces stay unchanged.
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=cfg["enable_thinking"],
            )
            rows.append(
                {
                    "input_ids": torch.tensor(ids),
                    "answer": example["answer"],
                    "question": example["question"],
                    "example_id": index,
                }
            )
        result.append(rows)
        selected[cfg[split_key]] = indices.tolist()
    if write_output:
        write_json(out / "selected_examples.json", selected)
    return result



def epoch_batches(train, cfg, state, rank=0, world_size=1):
    r"""Создать iterator оставшихся batches из явного состояния текущей epoch.
    
    Args:
        train: Готовые CPU prompt rows без случайных transforms.
        cfg: Per-rank batch size, workers и pin_memory settings.
        rank, world_size: Rank внутри single-node parallel run.
        state: Mutable loader checkpoint dict: epoch, order, next_batch,
            shuffle_rng, worker_rng и epoch_worker_rng.
    
    Returns:
        (iterator, global_epoch_update_count). Single GPU сохраняет partial tail;
        parallel shards равного размера, общий incomplete tail отбрасывается.
    
    Когда order отсутствует или epoch закончена, draw новой permutation использует
    saved shuffle generator и обновляет его state. Для частично обработанной epoch
    повторно создаётся только оставшийся batch_sampler. Worker seed воспроизводится
    из начала той же epoch; post-construction state сохраняется для следующей epoch.
    Prefetch не меняет next_batch: cursor двигает runner после optimizer update."""
    import torch
    from torch.utils.data import DataLoader

    local_size = cfg["prompt_batch_size"]
    size = local_size * world_size
    # Do not repeat examples to pad the last distributed batch. Every rank
    # receives the same local B; the shuffled incomplete tail is dropped.
    drop_last = world_size > 1 and cfg["distributed_drop_last"]
    count = len(train) // size if drop_last else (len(train) + size - 1) // size
    assert world_size == 1 or drop_last, "Parallel training requires equal, nonempty prompt shards"
    assert count > 0, "Train subset must contain at least one global prompt batch"
    if state["order"] is None or state["next_batch"] == count:
        shuffle_rng = torch.Generator().set_state(state["shuffle_rng"])
        state["order"] = torch.randperm(len(train), generator=shuffle_rng).tolist()
        state["shuffle_rng"] = shuffle_rng.get_state()
        state["epoch"] += 1
        state["next_batch"] = 0
        state["epoch_worker_rng"] = state["worker_rng"]
    # Prefetch never advances next_batch: only a completed optimizer update does.
    order = state["order"]
    batches = [
        order[i + rank*local_size : min(i + (rank+1)*local_size, len(order))]
        for i in range(state["next_batch"] * size, count*size, size)
    ]
    worker_rng = torch.Generator().set_state(state["epoch_worker_rng"])
    loader = DataLoader(
        train,
        batch_sampler=batches,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        collate_fn=collate,
        generator=worker_rng,
    )
    iterator = iter(loader)
    state["worker_rng"] = worker_rng.get_state()
    return iterator, count

