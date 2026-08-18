# Fine-tune pi0.5 on DexJoCo bimanual insertion

This integration targets the local LeRobot v3 dataset at
`/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly`.

It follows the DexJoCo π0.5 LoRA recipe (60k steps, batch 32, warmup 10k, flat
5e-5) on the official `pi05_base` checkpoint, with two dexterous-hand changes:

1. `DexJocoInputs` converts 46D quaternion proprioception to the 44D rotvec
   action layout so state and action share the same coordinates.
2. Convert official `pi05_base` offline to a 44D checkpoint with deterministic
   NNX Linear init on Allegro finger channels (`[6:22]`, `[28:44]`). Training
   loads that checkpoint with `CheckpointWeightLoader`.

```bash
JAX_PLATFORMS=cpu uv run python scripts/convert_pi05_dexjoco_action_head.py
```

Do not apply OpenPI `DeltaActions` here: subtracting rotvec is not a valid
SO(3) delta, and finger joints should stay absolute.

```bash
cd /home/wangrenpeng/openpi

# 1. Install without the optional RLDS dependency group.
uv sync

# 2. Compute quantile normalization statistics after the 46D -> 44D transform.
uv run scripts/compute_norm_stats.py --config-name pi05_dexjoco_lora

# 3. Paper-aligned LoRA run (default).
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_dexjoco_lora \
  --exp-name bimanual-insert-lora-v1 --overwrite

# 4. Optional full fine-tune. Override fsdp-devices to use multiple GPUs when free.
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/train.py pi05_dexjoco_full \
  --exp-name bimanual-insert-full-v1 --fsdp-devices 4 --overwrite
```

Serve and evaluate a checkpoint with the existing DexJoCo client. The server
applies the same 46D -> 44D transform, so the client can keep sending 46D
quaternion state.

```bash
# Terminal 1
cd /home/wangrenpeng/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_dexjoco_lora \
  --policy.dir /mnt/ssd/checkpoints/openpi_dexjoco/pi05_dexjoco_lora/bimanual-insert-lora-v1/60000

# Terminal 2
cd /home/wangrenpeng/dexjoco
conda run -n dexjoco env \
  PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco \
  MUJOCO_GL=egl \
  python -m dexjoco_openpi_client.cli.evaluate \
  --config configs/multi_task/bimanual_assembly.yaml \
  --host 127.0.0.1 --port 8000 --episodes 50 --seed 1000
```

For a fair comparison, evaluate checkpoints on identical held-out simulator
initial states and report success rate, insertion completion, time-to-success,
and contact/recovery failures. Do not select the final checkpoint on the same
seeds used for the headline result.
