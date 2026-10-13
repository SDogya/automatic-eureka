# Pretrained BD3LM benchmark on four A100 GPUs

```bash
git clone --branch pretrain-a100 --single-branch https://github.com/SDogya/automatic-eureka.git
cd automatic-eureka
bash pretrain_a100/run.sh --attempts 16 --gpus 0,1,2,3 --output "$HOME/bd3lm_benchmark"
```

The script installs its private environment, downloads the pinned Hugging Face
model and Kaggle GSM8K dataset, launches four pretrained inference workers in the
background, and archives responses, trajectories, TensorBoard and metrics.
Each attempt has its own reproducible seed. No TraFL training or adapters.

[Instructions, progress, resume and result format](pretrain_a100/README.md).

This branch includes the repository's earlier experiments; the A100 benchmark is
entirely contained in `pretrain_a100/` and does not use those experiments.
