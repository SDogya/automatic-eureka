"""One end-to-end CPU/Gloo check of the parallel training and resume paths."""
from datetime import timedelta
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import yaml

from trafl import training, evaluation
from trafl.checkpoint import restore_rng, save_checkpoint
from trafl.data import epoch_batches
from trafl.model import LogZ


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        positions = logits_to_keep if torch.is_tensor(logits_to_keep) else torch.arange(input_ids.shape[1])
        return SimpleNamespace(logits=self.weight * torch.arange(4.).expand(input_ids.shape[0], positions.numel(), 4))

    def get_base_model(self):
        return SimpleNamespace(model=lambda ids, **kwargs:
            SimpleNamespace(last_hidden_state=self.weight.expand(1, ids.shape[1], 2)))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / 'policy.pt')


def fresh():
    torch.manual_seed(42)
    policy, reference = TinyPolicy(), TinyPolicy()
    reference.requires_grad_(False)
    head = LogZ(2, 3, 'SiLU', True, True)
    with torch.no_grad():
        head.layers[-1].bias.fill_(.15)
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(head.parameters()), lr=.001, weight_decay=0.)
    return policy, reference, head, optimizer


def synthetic_group(model, prompt, count, *args):
    # Rank one has only EOS: its policy gradient is None before synchronization.
    ids = [[3, 0, 0]] * count if int(prompt[0]) % 2 else [[2, 2, 3], [1, 1, 3], [2, 2, 3], [1, 1, 3], [1, 1, 3]]
    return torch.tensor(ids)


def tokenizer():
    return SimpleNamespace(mask_token_id=3, eos_token_id=3, pad_token_id=0,
        batch_decode=lambda rows, **kwargs: [r'\boxed{2}' if r[0] == 2 else '' for r in rows])


def rows(count):
    return [dict(input_ids=torch.tensor([i, 0]), answer='#### 2', question=str(i), example_id=i) for i in range(count)]


def loader_state():
    return dict(epoch=-1, order=None, next_batch=0,
        shuffle_rng=torch.Generator().manual_seed(42).get_state(),
        worker_rng=torch.Generator().manual_seed(42).get_state())


class Writer:
    def add_scalar(self, *args):
        pass

    def flush(self):
        pass


def worker(rank, root, cfg):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous', rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    training.sample_group = evaluation.sample_group = synthetic_group
    policy, reference, head, optimizer = fresh()
    rng = torch.Generator().manual_seed(44 + rank*cfg['rank_seed_stride'])
    mask_rng = torch.Generator().manual_seed(43 + rank*cfg['rank_seed_stride'])
    state = loader_state()
    batches, count = epoch_batches(rows(9), cfg, state, rank, 2)
    shard = [row['example_id'] for batch in batches for row in batch]
    state['next_batch'] = 1
    remaining, _ = epoch_batches(rows(9), cfg, state, rank, 2)
    assert [r['example_id'] for b in remaining for r in b] == shard[1:]
    with (root / f'train_answers-rank{rank}.jsonl').open('w+') as record:
        stats = training.update_batch(policy, policy, reference, head, rows(2)[rank:rank+1], cfg,
            tokenizer(), rng, mask_rng, torch.bfloat16, torch.device('cpu'), optimizer, record)
        gradient = policy.weight.grad.clone()
        assert reference.weight.grad is None and reference.weight.item() == torch.tensor(.2).item()
        # Only the PEFT export interface is stubbed; real checkpoint/RNG/optimizer
        # serialization, gathering, atomic publication and replay run unchanged.
        import peft
        peft.get_peft_model_state_dict = lambda p: p.state_dict()
        save_checkpoint(policy, head, cfg, root, 1, optimizer, rng, mask_rng, state, stats, 1, record)
        saved = torch.load(root/'checkpoint/training_state.pt', weights_only=False)
        local = saved['rank_states'][rank]
        draws = torch.rand(4, generator=rng)
        restore_rng(local['rng'], rng, mask_rng)
        torch.testing.assert_close(torch.rand(4, generator=rng), draws, rtol=0, atol=0)
        restore_rng(local['rng'], rng, mask_rng)
        training.update_batch(policy, policy, reference, head, rows(2)[rank:rank+1], cfg,
            tokenizer(), rng, mask_rng, torch.bfloat16, torch.device('cpu'), optimizer, StringIO())
        continued = {k: v.clone() for k, v in policy.state_dict().items()}
        continued_head = {k: v.clone() for k, v in head.state_dict().items()}
        replay, ref, replay_head, replay_optimizer = fresh()
        replay.load_state_dict(saved['policy'])
        replay_head.load_state_dict(saved['head'])
        replay_optimizer.load_state_dict(saved['optimizer'])
        restore_rng(local['rng'], rng, mask_rng)
        training.update_batch(replay, replay, ref, replay_head, rows(2)[rank:rank+1], cfg,
            tokenizer(), rng, mask_rng, torch.bfloat16, torch.device('cpu'), replay_optimizer, StringIO())
        for key, value in replay.state_dict().items():
            torch.testing.assert_close(value, continued[key], rtol=0, atol=0)
        for key, value in replay_head.state_dict().items():
            torch.testing.assert_close(value, continued_head[key], rtol=0, atol=0)
    metrics = evaluation.evaluate(policy, policy, reference, head, rows(3), cfg, tokenizer(),
        torch.bfloat16, torch.device('cpu'), Writer() if rank == 0 else None, root, 2, rank, 2)
    torch.save(dict(policy=continued, head=continued_head, gradient=gradient, stats=stats,
                    shard=shard, order=state['order'], metrics=metrics), root/f'rank{rank}.pt')
    dist.destroy_process_group()


def test_parallel_update_matches_full_batch(tmp_path, monkeypatch):
    """Two ranks equal B=2; disjoint shards, empty gradients, resume and weighted eval."""
    cfg = yaml.safe_load(Path('config.yaml').read_text())
    cfg.update(num_workers=0, pin_memory=False, optimizer_fused=False, grad_clip=None,
        distributed_world_size=2, global_prompt_batch_size=2, updates=2,
        eval_samples=5, eval_pass_k=[1, 2, 5], eval_max_new_tokens=[3],
        score_completion_batch_size=2, score_microbatch=3)
    (tmp_path/'selected_examples.json').write_text(json.dumps({'train': [0, 1], 'test': [0, 1, 2]}))
    mp.spawn(worker, args=(str(tmp_path), cfg), nprocs=2, join=True)
    a, b = [torch.load(tmp_path/f'rank{i}.pt', weights_only=False) for i in range(2)]
    assert not set(a['shard']) & set(b['shard'])
    assert len(set(a['shard']+b['shard'])) == 8  # No repeated tail padding.
    assert set(a['shard']+b['shard']) == set(a['order'][:8])
    torch.testing.assert_close(a['gradient'], b['gradient'], rtol=0, atol=0)
    monkeypatch.setattr(training, 'sample_group', synthetic_group)
    monkeypatch.setattr(evaluation, 'sample_group', synthetic_group)
    policy, reference, head, optimizer = fresh()
    for step in range(2):
        stats = training.update_batch(policy, policy, reference, head, rows(2), cfg, tokenizer(),
            torch.Generator(), torch.Generator().manual_seed(43), torch.bfloat16,
            torch.device('cpu'), optimizer, StringIO())
        if step == 0:
            torch.testing.assert_close(policy.weight.grad, a['gradient'], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(stats, (a['stats']+b['stats'])/2, atol=2e-6, rtol=2e-6)
    for snapshot in (a, b):
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, snapshot['policy'][key], atol=2e-6, rtol=2e-6)
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, snapshot['head'][key], atol=2e-6, rtol=2e-6)
    expected_dir = tmp_path/'expected'
    expected_dir.mkdir()
    expected = evaluation.evaluate(policy, policy, reference, head, rows(3), cfg, tokenizer(),
        torch.bfloat16, torch.device('cpu'), Writer(), expected_dir, 2)
    for key, value in expected['3'].items():
        for result in (a, b):
            assert abs(result['metrics']['3'][key]-value) < 2e-6
    assert len(json.loads((tmp_path/'answers-step2-3.json').read_text())) == 3
    saved = torch.load(tmp_path/'checkpoint/training_state.pt', weights_only=False)
    assert len(saved['rank_states']) == 2
    assert saved['scheduler'] == dict(next_update=1, horizon=2)
    assert not torch.equal(saved['rank_states'][0]['rng']['rollout'], saved['rank_states'][1]['rng']['rollout'])
