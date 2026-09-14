# ABPP-XHand-Tube-Insert-from-tracker_collect_100：6 卡 / 20,000 steps 训练说明

## 训练目标

本文件用于管理 `ABPP-XHand-Tube-Insert-from-tracker_collect_100` 数据集的 Motus 训练。该数据集是 tracker 转换得到的 XHand 数据，配置使用 `action_dim=19`、`state_dim=18`，相机为 `front`、`left` 和 `wrist1`。代码中的 LeRobot loader 已支持这三个相机，因此本次只新增 6 卡 / 20,000 steps 的配置，不修改 XHand 的数据处理逻辑。

训练资源与保存位置如下：

| 内容 | 路径 |
|---|---|
| Motus 代码 | `/mnt/shared-storage-user/wanzhaoqi/Motus` |
| 本次配置 | `/mnt/shared-storage-user/wanzhaoqi/Motus/configs/xhand_tube_insert_tracker_6gpu_20k_wangyiqun.yaml` |
| 训练数据集 | `/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/ABPP-XHand-Tube-Insert-from-tracker_collect_100` |
| 预训练 WAN/VLM | `/mnt/shared-storage-user/wanzhaoqi/pretrained_models` |
| Motus 初始权重 | `/mnt/shared-storage-user/wanzhaoqi/pretrained_models/Motus` |
| checkpoint 根目录 | `/mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint` |
| W&B 离线日志 | `/mnt/shared-storage-user/wangyiqun/wanzhaoqi/wandb/xhand_tube_insert_tracker_20k_6gpu_bs2_wangyiqun` |

每张 GPU 的 batch size 为 2，6 卡有效 batch size 为 12。训练总步数和最终保存间隔均为 20,000。模型保存使用 `safe_serialization=True` 和 2 GB 分片，当前训练代码不调用 `accelerator.save_state`，因此不应生成 `pytorch_model*.bin` 或 `random_states*.pkl`。

下面四条命令按顺序执行。每个代码块都是一条独立命令；命令中的单引号只用于 `bash -lc` 或正则表达式，双引号用于路径和变量展开。

## 1. 配置文件与数据集检查

确认配置、LeRobot 元数据、数据文件、相机目录和维度均正确；同时打印 T5 任务数量的预期值。

```bash
CODE=/mnt/shared-storage-user/wanzhaoqi/Motus
CFG="$CODE/configs/xhand_tube_insert_tracker_6gpu_20k_wangyiqun.yaml"
DATA=/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/ABPP-XHand-Tube-Insert-from-tracker_collect_100

test -s "$CFG" && echo "配置文件 OK" || { echo "配置文件缺失"; exit 1; }
for p in "$DATA/meta/info.json" "$DATA/meta/tasks.parquet" "$DATA/meta/stats.json" "$DATA/data" "$DATA/videos"; do
  test -e "$p" && echo "OK: $p" || { echo "MISSING: $p"; exit 1; }
done

python3 -c "import json; d=json.load(open('$DATA/meta/info.json')); print('episodes=', d.get('total_episodes')); print('frames=', d.get('total_frames')); print('fps=', d.get('fps')); print('state=', d['features']['observation.state']['shape']); print('action=', d['features']['action']['shape']); print('cameras=', [k for k in d['features'] if k.startswith('observation.images.')]); print('tasks=', d.get('total_tasks'))"

grep -nE 'action_dim:|state_dim:|video_action_freq_ratio|task_name:|repo_id:|root:|batch_size:|max_steps:|checkpoint_dir:|save_interval:|run_name:' "$CFG"
```

## 2. 生成 T5 task embedding 缓存

该步骤只在数据集目录下生成 `t5_embedding/task_XXXXXX.pt`，不复制数据集；使用 wanzhaoqi 共享盘内的 Motus 环境和预训练模型，结果写入 wangyiqun 共享盘的数据集目录。

```bash
rjob submit \
  --name motus-xhand-tube-insert-tracker-t5-cache \
  --priority 9 \
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
set -e
source /mnt/shared-storage-user/wanzhaoqi/envs/motus/bin/activate
export PATH="/mnt/shared-storage-user/wanzhaoqi/envs/motus/bin:$PATH"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /mnt/shared-storage-user/wanzhaoqi/Motus
python data/lerobot/add_task_t5_cache_to_lerobot_dataset.py \
  --root /mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/ABPP-XHand-Tube-Insert-from-tracker_collect_100 \
  --wan_path /mnt/shared-storage-user/wanzhaoqi/pretrained_models \
  --device cuda \
  --text_len 512 \
  --t5_folder_name t5_embedding
'
```

## 3. 正式训练前检查

确认 T5 文件数量等于 `info.json` 的 `total_tasks`，确认 WAN/VLM/Motus 权重、GPU 和目标保存盘可用，并检查训练代码仍使用 safetensors 保存而不是完整 accelerate 状态保存。若目标 checkpoint 已存在 bin 或 pkl 文件，命令会停止，避免误用旧的完整状态目录。

```bash
CODE=/mnt/shared-storage-user/wanzhaoqi/Motus
CFG="$CODE/configs/xhand_tube_insert_tracker_6gpu_20k_wangyiqun.yaml"
DATA=/mnt/shared-storage-user/wangyiqun/AutoBioPlus/datasets/ABPP-XHand-Tube-Insert-from-tracker_collect_100
CKPT_BASE=/mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint
CKPT="$CKPT_BASE/xhand_tube_insert_tracker_6gpu_20k_wangyiqun"
T5="$DATA/t5_embedding"

test -s "$CFG" || { echo "配置文件缺失"; exit 1; }
test -f "$DATA/meta/info.json" || { echo "info.json 缺失"; exit 1; }
test -f "$DATA/meta/tasks.parquet" || { echo "tasks.parquet 缺失"; exit 1; }
test -f "$DATA/meta/stats.json" || { echo "stats.json 缺失"; exit 1; }
T5_COUNT=$(find "$T5" -maxdepth 1 -type f -name 'task_*.pt' -size +0c 2>/dev/null | wc -l)
EXPECTED_TASKS=$(python3 -c "import json; print(json.load(open('$DATA/meta/info.json')).get('total_tasks', 0))")
echo "T5 embedding: $T5_COUNT / $EXPECTED_TASKS"
test "$EXPECTED_TASKS" -gt 0 && test "$T5_COUNT" -eq "$EXPECTED_TASKS" || { echo "T5 embedding 数量不完整"; exit 1; }

for p in \
  /mnt/shared-storage-user/wanzhaoqi/pretrained_models/Wan2.2-TI2V-5B \
  /mnt/shared-storage-user/wanzhaoqi/pretrained_models/Qwen3-VL-2B-Instruct \
  /mnt/shared-storage-user/wanzhaoqi/pretrained_models/Motus; do
  test -e "$p" && echo "OK: $p" || { echo "MISSING: $p"; exit 1; }
done

if grep -nE 'accelerator\.save_state|pytorch_model|random_states' "$CODE/train/train.py"; then
  echo "发现不允许的完整状态保存调用或文件名"; exit 1
fi
grep -nE 'accelerator\.save_model|safe_serialization=True|max_shard_size' "$CODE/train/train.py"

if test -d "$CKPT" && find "$CKPT" -type f \( -name '*.bin' -o -name '*.pkl' \) -print -quit | grep -q .; then
  echo "目标 checkpoint 中已有 bin/pkl 文件，请先处理"; exit 1
fi

nvidia-smi -L
df -hT "$DATA" "$CKPT_BASE"
```

## 4. 提交 6 卡 / 20,000 steps 正式训练

该命令申请 6 张 GPU、64 核 CPU 和 160 GB 内存，优先级为 9。训练进程使用 `torch.distributed.run --nproc_per_node=6`，每 30 秒将 GPU 利用率写入 rjob 日志；W&B 采用 offline 模式保存到 wangyiqun 共享盘，不会覆盖其他 run。

```bash
rjob submit \
  --name motus-xhand-tube-insert-tracker-20k-6gpu \
  --priority 9 \
  --charged-group pceval_gpu \
  --restart-policy never \
  --preemptible no \
  --private-machine=group \
  --image registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:torchfinal \
  --cpu 64 \
  --gpu 6 \
  --memory 160000 \
  --mount gpfs://gpfs1/wanzhaoqi:/mnt/shared-storage-user/wanzhaoqi \
  --mount gpfs://gpfs1/wangyiqun:/mnt/shared-storage-user/wangyiqun \
  -- bash -lc '
set -e
source /mnt/shared-storage-user/wanzhaoqi/envs/motus/bin/activate
export PATH="/mnt/shared-storage-user/wanzhaoqi/envs/motus/bin:$PATH"
export WANDB_MODE=offline
export WANDB_PROJECT=motus
export WANDB_DIR=/mnt/shared-storage-user/wangyiqun/wanzhaoqi/wandb/xhand_tube_insert_tracker_20k_6gpu_bs2_wangyiqun
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p /mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint "$WANDB_DIR"
cd /mnt/shared-storage-user/wanzhaoqi/Motus
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 30 | sed "s/^/[GPU] /" &
GPU_MONITOR_PID=$!
set +e
python -m torch.distributed.run --nproc_per_node=6 train/train.py \
  --config configs/xhand_tube_insert_tracker_6gpu_20k_wangyiqun.yaml \
  --report_to wandb \
  --wandb_project motus \
  --run_name xhand_tube_insert_tracker_20k_6gpu_bs2_wangyiqun
TRAIN_STATUS=$?
kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
exit "$TRAIN_STATUS"
'
```

训练完成后，checkpoint 应位于 `/mnt/shared-storage-user/wangyiqun/wanzhaoqi/checkpoint` 下对应的 run 目录，W&B 离线文件位于上述 `WANDB_DIR`。如需查看进度，可用正式 job 返回的 job ID 执行 `rjob logs job <JOB_ID>`。
