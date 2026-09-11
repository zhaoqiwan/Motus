# ABPP-Wuji-Tweezers-Manipulate：临时 4 卡、30000 steps 训练说明

## 训练目标

本文件用于临时以 4 张 GPU、30000 steps 训练 `ABPP-Wuji-Tweezers-Manipulate`。数据集为
Wuji 三视角 LeRobot 数据，元数据为 20 FPS、100 个 episode、26 维 state、27 维 action，
相机键为 `front`、`left`、`wrist1`。Motus 当前 loader 已支持这组三相机，并会将三路
视频拼接成模型输入。

本文件是临时 4 卡版本，不覆盖原来的 2 卡/60000 steps 配置
`wuji_tweezers_manipulate_wangyiqun.yaml`。本次每卡 batch size 仍为 2，4 卡全局 batch
为 8；每 1000 steps 验证，在 30000 steps 保存最终 checkpoint。

## 路径约定

| 内容 | 路径 |
|---|---|
| 开发机代码 | `/mnt/shared-storage-user/wanzhaoqi/Motus` |
| Python 环境 | `/mnt/shared-storage-user/wanzhaoqi/envs/motus` |
| 预训练模型 | `/mnt/shared-storage-user/wanzhaoqi/pretrained_models` |
| 数据集 | `/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/autobioplus_lerobot/ABPP-Wuji-Tweezers-Manipulate` |
| 临时配置文件 | `/mnt/shared-storage-user/wanzhaoqi/Motus/configs/wuji_tweezers_manipulate_4gpu_wangyiqun.yaml` |
| 原 2 卡配置 | `/mnt/shared-storage-user/wanzhaoqi/Motus/configs/wuji_tweezers_manipulate_wangyiqun.yaml` |
| checkpoint 根目录 | `/mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint` |
| W&B 离线日志 | `/mnt/shared-storage-user/wangyiqun/wanzhaoqi/wandb` |

代码、环境和预训练模型从 `wanzhaoqi` 共享盘读取；数据集、T5 缓存、checkpoint 和 W&B
日志使用 `wangyiqun/wanzhaoqi` 共享盘。模型保存使用 safetensors，不保存 `.bin`、
optimizer 或随机状态文件。

以下四条指令均在开发机终端执行，并按顺序使用。

## 指令 1：配置文件和数据集检查

该指令不提交训练任务，只检查临时配置、数据集元数据、相机目录和存储空间。

```bash
CODE=/mnt/shared-storage-user/wanzhaoqi/Motus
DATA=/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/autobioplus_lerobot/ABPP-Wuji-Tweezers-Manipulate
CFG=$CODE/configs/wuji_tweezers_manipulate_4gpu_wangyiqun.yaml
READY=1

for f in "$CFG" "$DATA/meta/info.json" "$DATA/meta/tasks.parquet" "$DATA/meta/stats.json"; do
  if [ -s "$f" ]; then echo "OK: $f"; else echo "MISSING: $f"; READY=0; fi
done
for d in "$DATA/data" "$DATA/meta/episodes" "$DATA/videos/observation.images.front" "$DATA/videos/observation.images.left" "$DATA/videos/observation.images.wrist1"; do
  if [ -d "$d" ]; then echo "OK: $d"; else echo "MISSING: $d"; READY=0; fi
done

echo "=== 数据集元数据 ==="
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); print({"total_episodes":x["total_episodes"],"total_frames":x["total_frames"],"total_tasks":x["total_tasks"],"fps":x["fps"],"state_dim":x["features"]["observation.state"]["shape"][0],"action_dim":x["features"]["action"]["shape"][0]})' "$DATA/meta/info.json"
grep -o '"observation.images[^"]*"' "$DATA/meta/info.json" | sort -u

echo "=== 配置关键字段 ==="
grep -nE 'action_dim|state_dim|task_name|repo_id|root:|batch_size|max_steps|checkpoint_dir|save_interval|val_interval|run_name' "$CFG"
du -sh "$DATA"
df -hT /mnt/shared-storage-user/wanzhaoqi /mnt/shared-storage-user/wangyiqun
if [ "$READY" -eq 1 ]; then echo "基础检查通过"; else echo "基础检查未通过"; fi
```

## 指令 2：生成 T5 embedding

该任务使用 1 张 GPU，只在现有数据集目录生成任务级 T5 缓存，不复制原始数据集。

```bash
rjob submit \
  --name motus-wuji-tweezers-t5-cache-wangyiqun \
  --charged-group pceval_gpu \
  --restart-policy never \
  --preemptible no \
  --private-machine=group \
  --image registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:torchfinal \
  --cpu 8 \
  --gpu 1 \
  --memory 32000 \
  --mount gpfs://gpfs1/wanzhaoqi:/mnt/shared-storage-user/wanzhaoqi \
  --mount gpfs://gpfs1/wangyiqun:/mnt/shared-storage-user/wangyiqun \
  -- bash -lc '
set -euo pipefail
source /mnt/shared-storage-user/wanzhaoqi/envs/motus/bin/activate
export PATH="/mnt/shared-storage-user/wanzhaoqi/envs/motus/bin:$PATH"
cd /mnt/shared-storage-user/wanzhaoqi/Motus
python data/lerobot/add_task_t5_cache_to_lerobot_dataset.py \
  --root /mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/autobioplus_lerobot/ABPP-Wuji-Tweezers-Manipulate \
  --wan_path /mnt/shared-storage-user/wanzhaoqi/pretrained_models \
  --device cuda
'
```

## 指令 3：正式训练前检查

T5 任务显示 `Succeeded` 后执行。该数据集只有 1 个任务，因此必须存在且只有一个非空
`task_*.pt` 文件。所有检查通过后才可提交指令 4。

```bash
CODE=/mnt/shared-storage-user/wanzhaoqi/Motus
DATA=/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/autobioplus_lerobot/ABPP-Wuji-Tweezers-Manipulate
CFG=$CODE/configs/wuji_tweezers_manipulate_4gpu_wangyiqun.yaml
TRAIN=$CODE/train/train.py
PRE=/mnt/shared-storage-user/wanzhaoqi/pretrained_models
READY=1

FOUND_TASKS=$(find "$DATA/t5_embedding" -maxdepth 1 -type f -name 'task_*.pt' -size +0c 2>/dev/null | wc -l)
echo "T5 缓存: $FOUND_TASKS/1"
if [ "$FOUND_TASKS" -ne 1 ]; then READY=0; fi

for f in "$PRE/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" "$PRE/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"; do
  if [ -s "$f" ]; then echo "OK: $f"; else echo "MISSING: $f"; READY=0; fi
done
for d in "$PRE/Wan2.2-TI2V-5B" "$PRE/Qwen3-VL-2B-Instruct" "$PRE/Motus"; do
  if [ -d "$d" ]; then echo "OK: $d"; else echo "MISSING: $d"; READY=0; fi
done

grep -q 'action_dim: 27' "$CFG" || READY=0
grep -q 'state_dim: 26' "$CFG" || READY=0
grep -q 'task_name: "ABPP-Wuji-Tweezers-Manipulate"' "$CFG" || READY=0
grep -q 'repo_id: "ABPP-Wuji-Tweezers-Manipulate"' "$CFG" || READY=0
grep -q 'max_steps: 30000' "$CFG" || READY=0
grep -q 'save_interval: 30000' "$CFG" || READY=0
grep -q 'checkpoint_dir: "/mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint"' "$CFG" || READY=0
grep -q 'run_name: "wuji_tweezers_manipulate_30k_4gpu_bs2_wangyiqun"' "$CFG" || READY=0
grep -nE 'action_dim|state_dim|task_name|repo_id|root:|batch_size|max_steps|checkpoint_dir|save_interval|run_name' "$CFG"

grep -n 'safe_serialization=True' "$TRAIN" || { echo "ERROR: 未找到 safetensors 保存"; READY=0; }
if grep -nE 'save_state|pytorch_model|random_states' "$TRAIN"; then
  echo "ERROR: 发现可能生成 bin 或训练状态文件的逻辑"
  READY=0
else
  echo "OK: 未发现 bin/optimizer/random-state 保存逻辑"
fi

if [ ! -w /mnt/shared-storage-user/wangyiqun/wanzhaoqi ]; then READY=0; fi
df -hT /mnt/shared-storage-user/wanzhaoqi /mnt/shared-storage-user/wangyiqun
if [ "$READY" -eq 1 ]; then echo "READY: 可以正式训练"; else echo "NOT READY: 不要提交训练"; fi
```

## 指令 4：正式训练

该任务使用 4 张 GPU、64 核 CPU、30000 steps 和优先级 9。临时 4 卡任务使用独立的
run name 和 W&B 目录，不会覆盖 2 卡/60000 steps 的记录。

```bash
rjob submit \
  --name motus-wuji-tweezers-30k-4gpu-wangyiqun \
  --priority 9 \
  --charged-group pceval_gpu \
  --restart-policy never \
  --preemptible no \
  --private-machine=group \
  --image registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:torchfinal \
  --cpu 64 \
  --gpu 4 \
  --memory 160000 \
  --mount gpfs://gpfs1/wanzhaoqi:/mnt/shared-storage-user/wanzhaoqi \
  --mount gpfs://gpfs1/wangyiqun:/mnt/shared-storage-user/wangyiqun \
  -- bash -lc '
set -euo pipefail
source /mnt/shared-storage-user/wanzhaoqi/envs/motus/bin/activate
export PATH="/mnt/shared-storage-user/wanzhaoqi/envs/motus/bin:$PATH"
export WANDB_MODE=offline
export WANDB_PROJECT=motus
export WANDB_DIR=/mnt/shared-storage-user/wangyiqun/wanzhaoqi/wandb/wuji_tweezers_manipulate_30k_4gpu_bs2
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

CODE=/mnt/shared-storage-user/wanzhaoqi/Motus
DATA=/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/autobioplus_lerobot/ABPP-Wuji-Tweezers-Manipulate
FOUND_TASKS=$(find "$DATA/t5_embedding" -maxdepth 1 -type f -name "task_*.pt" -size +0c | wc -l)
if [ "$FOUND_TASKS" -ne 1 ]; then echo "ERROR: T5 缓存数量错误"; exit 1; fi

cd "$CODE"
mkdir -p /mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint "$WANDB_DIR"
grep -q "safe_serialization=True" train/train.py
if grep -qE "save_state|pytorch_model|random_states" train/train.py; then
  echo "ERROR: 检测到可能生成 bin 的保存逻辑"
  exit 1
fi

exec python -m torch.distributed.run \
  --nproc_per_node=4 \
  train/train.py \
  --config configs/wuji_tweezers_manipulate_4gpu_wangyiqun.yaml \
  --report_to wandb \
  --wandb_project motus \
  --run_name wuji_tweezers_manipulate_30k_4gpu_bs2_wangyiqun
'
```
