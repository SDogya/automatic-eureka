"""Measure real BD3LM forwards; compare scores and exercise one joint update.

CUDA_VISIBLE_DEVICES selects the probe GPU. --dtype float16 permits measurements
on the older server cards; production A100 configuration remains BF16. No model
is downloaded, no dataset is altered, and no checkpoint is overwritten.
"""
import argparse
from io import StringIO
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
import dllm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import snapshot_download
from trafl.cached_sampling import CachedSampler
from trafl.masking import antithetic_masks
from trafl.model import LogZ
from trafl.sampling import generate
from trafl.scoring import masked_score, masked_scores
from trafl.training import update_batch

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml')
parser.add_argument('--dtype', default='bfloat16')
parser.add_argument('--output', default='performance_probe.json')
args = parser.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
dtype = getattr(torch, args.dtype)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.manual_seed(cfg['seed'])
path = snapshot_download(cfg['model_id'], revision=cfg['model_revision'], local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(path, dtype=dtype, attn_implementation='sdpa').cuda().eval()
reference = AutoModelForMaskedLM.from_pretrained(path, dtype=dtype, attn_implementation='sdpa').cuda().eval()
reference.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(path)
model = get_peft_model(model, LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
    target_modules=cfg['lora_targets'], lora_dropout=cfg['lora_dropout'], bias=cfg['lora_bias']))
model.eval()
prompt = torch.tensor(tokenizer.encode('Solve 2+2. Put the final answer in \\boxed{}.'), device='cuda')
# Short real-model probe fits alongside existing jobs. Group and mask counts
# remain the actual training G=5, K=4, with the full rank-128 adapters.
length = 32
rng = lambda: torch.Generator(device='cuda').manual_seed(44)
sampler = CachedSampler(model, cfg)

def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter()-begin, torch.cuda.max_memory_allocated()/2**30

old_call = lambda: generate(model, prompt, 5, length, 32, 0., .91, .9, tokenizer.mask_token_id,
                            True, rng(), dtype)
new_call = lambda: sampler.sample(prompt, 5, length, 32, 0., .91, .9, tokenizer.mask_token_id,
                            True, rng(), dtype, tokenizer.eos_token_id, False)
old_call()  # Warm CUDA kernels before comparing timings.
old, old_seconds, old_mem = timed(old_call)
new, cold_seconds, cold_mem = timed(new_call)
new, new_seconds, new_mem = timed(new_call)
assert torch.equal(old, new), 'Greedy cached and full-canvas tokens differ'
# Compare the actual stochastic training protocol after the equivalence check.
old_call = lambda: generate(model, prompt, 5, length, 32, cfg['train_temperature'], .91, .9,
                            tokenizer.mask_token_id, True, rng(), dtype)
new_call = lambda: sampler.sample(prompt, 5, length, 32, cfg['train_temperature'], .91, .9,
                            tokenizer.mask_token_id, True, rng(), dtype, tokenizer.eos_token_id, False)
old_call()
old, old_seconds, old_mem = timed(old_call)
new, cold_seconds, cold_mem = timed(new_call)
new, new_seconds, new_mem = timed(new_call)
completions = list(new.unbind())
mask_rng = torch.Generator().manual_seed(43)
masks = [antithetic_masks((x.cpu()!=tokenizer.eos_token_id)&(x.cpu()!=tokenizer.pad_token_id),
                         4, mask_rng, cfg['mask_endpoint']) for x in completions]
with torch.no_grad():
    scalar, scalar_seconds, _ = timed(lambda: torch.stack([masked_score(model, prompt, x, m,
        tokenizer.mask_token_id, dtype, 0., 32) for x, m in zip(completions, masks)]))
    batched, batched_seconds, _ = timed(lambda: masked_scores(model, prompt, completions, masks,
        tokenizer.mask_token_id, dtype, 0., 32, tokenizer.pad_token_id, 4))
torch.testing.assert_close(scalar, batched, atol=.02, rtol=.005)
# Use conservative chunks for this 8GB server; A100 defaults batch all 20 scores.
cfg.update(score_completion_batch_size=1, score_microbatch=1, train_max_new_tokens=8,
           generation_compile=False, generation_cuda_graph=False, optimizer_fused=True)
from trafl import scoring
scoring.batched_token_scores = torch.compile(scoring.batched_token_scores, backend=cfg['compile_backend'], dynamic=True)
head = LogZ(model.config.hidden_size, cfg['head_hidden'], 'SiLU', True, True).cuda()
# Nonzero head mimics an already-trained state and ensures a nonzero residual
# even when all five pretrained answers receive the same binary reward.
with torch.no_grad():
    head.layers[-1].bias.fill_(.1)
parameters = [p for p in list(model.parameters())+list(head.parameters()) if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=cfg['lr_peak'], weight_decay=0., fused=True)
rows = [dict(input_ids=prompt[:8], answer='#### 4', question='2+2?', example_id=0)]
model._trafl_sampler = CachedSampler(model, cfg)
stats, update_seconds, update_mem = timed(lambda: update_batch(model, model, reference, head, rows,
    cfg, tokenizer, rng(), torch.Generator().manual_seed(43), dtype, torch.device('cuda'), optimizer, StringIO()))
assert torch.isfinite(stats).all()
assert all(p.grad is None for p in reference.parameters())
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n)
report = dict(model_id=cfg['model_id'], revision=cfg['model_revision'], torch=torch.__version__,
    gpu=torch.cuda.get_device_name(), dtype=args.dtype, prompt_tokens=prompt.numel(), completion_tokens=length,
    rollouts=5, masks=4, temperature=cfg['train_temperature'], top_p=.91, greedy_tokens_equal=True, score_max_difference=(scalar-batched).abs().max().item(),
    rollout_before_seconds=old_seconds, rollout_after_cold_seconds=cold_seconds,
    rollout_after_warm_seconds=new_seconds, rollout_speedup=old_seconds/new_seconds,
    score_before_seconds=scalar_seconds, score_after_seconds=batched_seconds,
    score_speedup=scalar_seconds/batched_seconds, joint_update_seconds=update_seconds,
    joint_update_peak_gib=update_mem, joint_update_stats=stats.cpu().tolist(),
    frozen_reference=True, lora_gradients_nonzero=True)
Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
