# 复现操作说明

本文档说明如何从一台没有配置过本项目环境的 Linux 机器开始，完成环境安装、数据路径配置、训练/续训 checkpoint，以及使用最终 checkpoint 生成提交结果。命令默认在仓库根目录执行，也就是包含 `stage1/`、`stage2/`、`stage3/` 的 `code/` 目录：

```bash
cd /path/to/code
export CODE_ROOT="$(pwd)"
```

除非特别说明，下面所有路径都写成相对 `CODE_ROOT` 的形式。复现时只需要把数据根目录和 checkpoint 路径替换成自己机器上的实际位置。

## 1. 项目改动概览

本版本是在原提交代码基础上继续改动得到的，主要变化如下：

1. 在第二次去噪 PGD2 前向中加入局部 PCA 法向坐标系。推理时先对 PGD1 输出估计 patch 局部法向，将点云转到 tangent/tangent/normal 坐标系后执行 PGD2，再旋回世界坐标。
2. 修改最终推理逻辑，使用高覆盖 overlapping patch、adaptive merge、local-alpha、Surface Gate 和 Coverage Repair 形成单入口推理流程。
3. 调整 Stage3 的 hard-tail loss、学习率、batch size 和训练 patch 数，并从已有 checkpoint 分段续训，最终使用 `stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl` 推理。

最终推理不会读取测试集 GT、mesh 或外部 normal；法向均由输入点云/中间点云局部 PCA 在线估计。

## 2. 环境安装

### 2.1 推荐软硬件

```text
Ubuntu 22.04 或兼容 Linux
NVIDIA GPU，训练和最终单卡推理建议至少 24GB 显存
NVIDIA 驱动需兼容 CUDA 12.2 及以上版本
CUDA Toolkit 推荐 12.4
Python 3.9 或 3.10
g++、nvcc 可用
```

本项目历史稳定环境：

```text
Python 3.9
Jittor 1.3.11.0
NumPy 1.26.4
SciPy 1.13.1
OmegaConf 2.3.1
```

不要使用 Python 3.12 的 base 环境运行 Jittor。Jittor 首次运行会即时编译 C++/CUDA 算子，需要等待几分钟到十几分钟，并确保缓存目录有写入权限。

### 2.2 创建 Conda 环境

推荐手工创建一个独立环境：

```bash
conda create -n pgd_jt python=3.9 pip -y
conda activate pgd_jt
python -m pip install --upgrade pip

python -m pip install \
  jittor==1.3.11.0 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  omegaconf==2.3.1 \
  PyYAML==6.0.3 \
  tqdm==4.68.3 \
  astunparse==1.6.3 \
  antlr4-python3-runtime==4.9.3 \
  six==1.17.0
```

也可以直接使用根目录环境文件：

```bash
conda env create -f environment.yaml
conda activate pgd_jt
python -m pip install -r requirements.txt
```

`point-cloud-utils` 和 `trimesh` 主要用于可选的 Surface Gate 训练数据构建。只运行最终推理时，核心依赖是 Jittor、NumPy、SciPy、OmegaConf 和 PyYAML。

### 2.3 CUDA 和运行前环境变量

如果机器安装了 CUDA 12.4，可按实际安装位置设置：

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

每次训练或推理前建议设置：

```bash
export CUDA_VISIBLE_DEVICES=0
export cc_path="$(which g++)"
export HWLOC_COMPONENTS=-gl
export OBJ_MESH_CACHE_MB=512
```

如果数据加载异常、卡死或机器 CPU 资源紧张，可把对应 `configs/data/train.yaml` 中的 `num_workers` 临时改为 `0` 后重试。

## 3. 数据准备和路径修改

### 3.1 数据目录格式

训练数据根目录应包含官方训练样本，样本内通常有：

```text
<DATA_ROOT>/dataset_train/shapenet/<category>/<shape_id>/models/model_normalized.obj
```

测试 noisy 数据根目录应包含：

```text
<DATA_ROOT>/dataset_test_noisy/dataset_test_noisy/shapenet/<category>/<sample_id>/noisy.npy
```

例如可以在复现机器上组织成：

```bash
export DATA_ROOT=/path/to/jittor_dataset
```

对应：

```text
$DATA_ROOT/dataset_train
$DATA_ROOT/dataset_test_noisy/dataset_test_noisy
```

`datalist/*.txt` 每行保存相对于数据根目录的样本路径。例如 B 榜测试列表 `stage3/datalist/test.txt` 中的每行应能和 `--input-root` 拼成：

```text
<input-root>/<entry>/noisy.npy
```

### 3.2 修改训练数据根目录

三个训练阶段的数据根目录都集中在各自的 data YAML 中：

```text
stage1/configs/data/train.yaml
stage2/configs/data/train.yaml
stage3/configs/data/train.yaml
```

将其中的 `input_dataset_dir` 改为本机训练集路径：

```yaml
datapath:
  input_dataset_dir: /path/to/jittor_dataset/dataset_train
```

Stage1 和 Stage2 分别有 `train_dataset`、`validate_dataset` 两处 `input_dataset_dir`；Stage3 当前配置主要有 `train_dataset`。如果你的验证集也放在同一训练根目录，三处都写同一个路径即可。

### 3.3 修改训练 list

训练链中使用过三类 list：

```text
A 榜训练集
A+B 完整训练集
A 的部分数据 + B 训练集
```

当前仓库已经整理好本次复现使用的 list：

```text
stage1/datalist/train.txt
stage1/datalist/validate.txt
stage2/datalist/train.txt
stage2/datalist/validate.txt
stage3/datalist/train.txt
stage3/datalist/test.txt
```

如果你重新放置或重新划分数据，只需要保持 list 中的相对路径和 `input_dataset_dir` 能拼到真实文件。例如训练样本最终应能找到：

```text
<input_dataset_dir>/<list_entry>/models/model_normalized.obj
```

测试样本最终应能找到：

```text
<input_root>/<test_list_entry>/noisy.npy
```

## 4. 代码结构

仓库按 `stage1 -> stage2 -> stage3` 的训练链组织。三个阶段的 `src/` 目录结构基本一致，区别主要在训练入口和 task 配置。

```text
code/
  REPRODUCTION_GUIDE.md
    本文档。

  ckp/
    复现链路保留的关键 checkpoint。最终推理默认使用这里的 544 权重。

  environment.yaml
  requirements.txt
    Conda 环境和 pip 依赖清单。

  stage1/
    run.py
      Stage1 训练入口，训练单阶段 PGD1。
    configs/
      data/train.yaml
      task/train.yaml
      task/train_from_checkpoint_349_200.yaml
      model/core.yaml
      transform/ops.yaml
      system/run.yaml
    datalist/
      train.txt、validate.txt、test.txt
    src/
      data/、model/、system/、utils/
    third_party/PointCloudLib/
      仓库内置 KNN/group 算子。

  stage2/
    run.py
      Stage2 训练入口，加载 PGD1 并训练 PGD2。
    configs/
      data/train.yaml
      task/freeze40.yaml
      model/core.yaml
      transform/ops.yaml
      system/run.yaml
    datalist/
    src/
    third_party/PointCloudLib/

  stage3/
    run.py
      Stage3 联合训练入口，加载 Stage1/Stage2 或 joint checkpoint。
    final_inference.py
      最终推理入口，包含高覆盖 patch 前向、PGD2 法向坐标系、local-alpha、
      Surface Gate、Coverage Repair 和 zip 打包。
    compare_prediction_dirs.py
      复现结果与参考结果的逐点对比脚本。
    configs/
      task/joint.yaml
      task/joint_from_200_hard005_tail015_bs32.yaml
      task/joint_from_362_hard005_tail015_bs16_fps_np2_100.yaml
      task/joint_from_461_hard005_tail015_bs8_fps_np4_100.yaml
      transform/ops.yaml、ops_fps_np2.yaml、ops_fps_np4.yaml
      system/run*.yaml
      model/core.yaml
      data/train.yaml
    datalist/
      train.txt、validate.txt、test.txt
    models/surface_gate/
      最终后处理使用的五折 Gate 权重。
    src/
      data/、model/、system/、utils/
    third_party/PointCloudLib/
```

复现最终结果的最小核心文件是：

```text
stage3/final_inference.py
stage3/src/
stage3/configs/model/core.yaml
stage3/configs/transform/ops.yaml
stage3/third_party/PointCloudLib/
stage3/models/surface_gate/
stage3/datalist/test.txt
ckp/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl
```

如果还要复现完整训练链，还需要保留：

```text
stage1/
stage2/
stage3/run.py
stage3/configs/task/joint.yaml
stage3/configs/task/joint_from_200_hard005_tail015_bs32.yaml
stage3/configs/task/joint_from_362_hard005_tail015_bs16_fps_np2_100.yaml
stage3/configs/task/joint_from_461_hard005_tail015_bs8_fps_np4_100.yaml
stage3/configs/transform/ops_fps_np2.yaml
stage3/configs/transform/ops_fps_np4.yaml
ckp/stage1_checkpoint_349.pkl
ckp/stage1_checkpoint_199.pkl
ckp/stage2_checkpoint_39.pkl
ckp/stage3_checkpoint_joint_200.pkl
ckp/stage3_checkpoint_hard005_tail015_bs32_joint_362.pkl
ckp/stage3_checkpoint_hard005_tail015_bs16_fps_np2_from362_joint_461.pkl
```

## 5. 最终推理和提交文件生成

如果只需要复现最终提交结果，可以先跳过训练复现，直接使用仓库 `ckp/` 中保存的 544 checkpoint：

```text
ckp/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl
```

最终提交推理入口是：

```text
stage3/final_inference.py
```

该脚本执行：

```text
noisy.npy
  -> PGD1/PGD2 overlapping patch forward
  -> 保存 pgd1.npy、pgd2_raw.npy、原始 denoised.npy
  -> local-alpha
  -> Surface Gate
  -> Coverage Repair
  -> final denoised.npy
  -> submission zip
```

Surface Gate 五折权重位于：

```text
stage3/models/surface_gate/gate_fold0.npz
stage3/models/surface_gate/gate_fold1.npz
stage3/models/surface_gate/gate_fold2.npz
stage3/models/surface_gate/gate_fold3.npz
stage3/models/surface_gate/gate_fold4.npz
```

### 5.1 单卡 B 榜推理命令

先设置测试数据位置：

```bash
export TEST_ROOT=/path/to/jittor_dataset/dataset_test_noisy/dataset_test_noisy
```

使用 `ckp/` 中的 544 权重推理：

```bash
python stage3/final_inference.py \
  --checkpoint ckp/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl \
  --input-root "$TEST_ROOT" \
  --test-list stage3/datalist/test.txt \
  --work-dir stage3/final_runs/544_b_200 \
  --output-zip stage3/final_runs/result_544_b_200.zip \
  --gpu-id 0 \
  --post-workers 16
```

如果使用自己续训得到的最终 checkpoint，把 `--checkpoint` 改为：

```text
stage3/experiments_hard005_tail015_bs8_fps_np4_from461/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl
```

### 5.2 推理默认稳定参数

这些参数已经是 `final_inference.py` 默认值，通常不需要手动传入：

```text
patch_size                 = 2600
patch_seed_k               = 52
patch_seed_k_alpha         = 10
patch_merge                = adaptive
patch_merge_beta           = 8.0
patch_adaptive_alpha_max   = 1.0
patch_adaptive_tau         = 0.008
normal_frame_sign_mode     = stable_dominant
tta_passes                 = 1

local-alpha:
  k                         = 16
  base                      = 1.20
  delta                     = 0.07
  band                      = [0.05, 0.25]

Surface Gate:
  neighborhood scales       = [8, 12, 16, 24, 32]
  basis weights             = [0.2, 0.2, 0.2, 0.2, 0.2]
  gate_scale                = 0.575

Coverage Repair:
  k                         = 16
  coverage_strength         = 0.90
  max_step_ratio            = 0.01
```

### 5.3 输出目录和提交 zip

推理完成后会得到：

```text
stage3/final_runs/544_b_200/
  run_manifest.json
  intermediates/<sample>/pgd1.npy
  intermediates/<sample>/pgd2_raw.npy
  intermediates/<sample>/denoised.npy
  local_alpha/<sample>/denoised.npy
  final/<sample>/denoised.npy

stage3/final_runs/result_544_b_200.zip
```

提交 zip 中只包含最终文件：

```text
shapenet/<category>/<sample-id>/denoised.npy
```

提交前可检查压缩包：

```bash
unzip -tq stage3/final_runs/result_544_b_200.zip
```

### 5.4 断点续跑和分阶段运行

默认不覆盖已有完整文件。推理中断后，重新执行同一命令即可续跑；已有的 `intermediates/`、`local_alpha/`、`final/` 文件会被跳过。

强制全部重跑：

```bash
python stage3/final_inference.py ... --overwrite
```

分阶段运行：

```bash
# 只做 GPU forward，保存 pgd1.npy、pgd2_raw.npy 和原始 denoised.npy
python stage3/final_inference.py ... --mode forward

# 读取已有 intermediates，只做 CPU 后处理
python stage3/final_inference.py ... --mode postprocess

# 只校验 final 并打包 zip
python stage3/final_inference.py ... --mode package
```

`--post-workers` 是 CPU 后处理进程数，不是 GPU 进程数。单卡 4090 不需要 `mpirun`，也不要在同一张卡上同时启动多个最终推理进程。

## 6. 训练和续训流程

完整训练链如下：

```text
1. Stage1: A 榜数据训练 500 轮
2. Stage1: 从 A 榜第 349/350 轮 checkpoint 在 A+B 完整数据上续训 200 轮
3. Stage2: 加载 Stage1 续训后的 199.pkl，训练 PGD2 40 轮
4. Stage3: 加载 Stage1 199.pkl + Stage2 39.pkl，联合训练到 joint_200.pkl
5. Stage3: 从 joint_200.pkl 用 hard005 tail015 bs32 续训到 400 轮附近，选 362.pkl
6. Stage3: 从 362.pkl 改为 num_patches=2、bs16 续训到 461.pkl
7. Stage3: 从 461.pkl 改为 num_patches=4、bs8 续训到最终 544.pkl
```

仓库根目录的 `ckp/` 已保存关键 checkpoint，可用于直接复现后续阶段或最终推理：

```text
ckp/stage1_checkpoint_349.pkl
ckp/stage1_checkpoint_199.pkl
ckp/stage2_checkpoint_39.pkl
ckp/stage3_checkpoint_joint_200.pkl
ckp/stage3_checkpoint_hard005_tail015_bs32_joint_362.pkl
ckp/stage3_checkpoint_hard005_tail015_bs16_fps_np2_from362_joint_461.pkl
ckp/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl
```

如果从零训练，请把各 task YAML 中的 `load_ckpt`、`stage1_ckpt`、`stage2_ckpt`、`init_joint_ckpt` 改为上一阶段实际输出的路径。如果使用 `ckp/` 中的现成权重，也可以把 YAML 中对应字段指向 `ckp/<checkpoint_name>.pkl`。

### 6.1 Stage1: A 榜数据训练 500 轮

```bash
python stage1/run.py --task stage1/configs/task/train
```

主要配置：

```text
stage1/configs/task/train.yaml
stage1/configs/data/train.yaml
stage1/configs/model/core.yaml
stage1/configs/transform/ops.yaml
stage1/configs/system/run.yaml
```

关键参数：

```yaml
trainer:
  epochs: 500
optimizer:
  lr: 0.0005
train_dataset:
  batch_size: 32
model:
  patch_size: 1000
  niters: 1
  seed_k: 6
  seed_k_alpha: 10
```

输出：

```text
stage1/experiments/stage1_checkpoint_<epoch>.pkl
```

后续续训通常使用第 349 轮 checkpoint，即 `stage1_checkpoint_349.pkl`。

### 6.2 Stage1: A+B 完整数据续训 200 轮

先确认 `stage1/configs/data/train.yaml` 对应的是 A+B 完整训练数据 list，然后执行：

```bash
python stage1/run.py --task stage1/configs/task/train_from_checkpoint_349_200
```

关键配置：

```yaml
load_ckpt: ckp/stage1_checkpoint_349.pkl
trainer:
  epochs: 200
optimizer:
  lr: 0.0005
```

如果 YAML 中仍是旧路径，请改成上一步实际 checkpoint。输出仍在：

```text
stage1/experiments/
```

后续使用：

```text
stage1/experiments/stage1_checkpoint_199.pkl
```

### 6.3 Stage2: 加载 Stage1 199.pkl 训练 PGD2

```bash
python stage2/run.py --task stage2/configs/task/freeze40
```

关键配置：

```yaml
stage1_ckpt: ckp/stage1_checkpoint_199.pkl
epochs: 40
lr: 0.00005
lambda_hard: 0.002
hard_tail_ratio: 0.1
grad_clip: 1.0
```

如从零训练，请把 `stage1_ckpt` 改成 `stage1/experiments/stage1_checkpoint_199.pkl`。输出：

```text
stage2/experiments/stage2_checkpoint_39.pkl
```

### 6.4 Stage3: A+B 联合训练到 joint_200.pkl

```bash
python stage3/run.py --task stage3/configs/task/joint
```

关键配置：

```yaml
stage1_ckpt: ckp/stage1_checkpoint_199.pkl
stage2_ckpt: ckp/stage2_checkpoint_39.pkl
joint_epochs: 200
joint_batch_size: 16
stage2_lr: 0.00005
stage1_lr_mult: 0.1
stage1_loss_weight: 0.1
lambda_hard: 0.002
hard_tail_ratio: 0.1
grad_clip: 1.0
```

输出：

```text
stage3/experiments/stage3_checkpoint_joint_200.pkl
```

### 6.5 Stage3: hard005 tail015 bs32 续训

这一阶段从 `joint_200.pkl` 开始，增强 hard-tail loss，并使用 A 的部分数据 + B 训练集：

```bash
python stage3/run.py \
  --task stage3/configs/task/joint_from_200_hard005_tail015_bs32
```

关键配置：

```yaml
init_joint_ckpt: ckp/stage3_checkpoint_joint_200.pkl
start_epoch: 201
joint_epochs: 400
joint_batch_size: 32
stage2_lr: 0.00005
stage1_lr_mult: 0.1
stage1_loss_weight: 0.1
lambda_hard: 0.005
hard_tail_ratio: 0.15
```

输出目录：

```text
stage3/experiments_hard005_tail015_bs32/
```

本链路选择本地评估表现最好的 362 checkpoint 进入下一阶段：

```text
stage3/experiments_hard005_tail015_bs32/stage3_checkpoint_hard005_tail015_bs32_joint_362.pkl
```

### 6.6 Stage3: num_patches=2、bs16 续训到 461

这一阶段将训练 patch 数从 1 改为 2，并使用 FPS 风格的 patch center 采样。24GB 显存下 bs32/bs24 会 OOM，bs16 可以启动训练。

```bash
python stage3/run.py \
  --task stage3/configs/task/joint_from_362_hard005_tail015_bs16_fps_np2_100
```

关键配置：

```yaml
init_joint_ckpt: ckp/stage3_checkpoint_hard005_tail015_bs32_joint_362.pkl
start_epoch: 363
joint_epochs: 462
joint_batch_size: 16
lambda_hard: 0.005
hard_tail_ratio: 0.15
components:
  transform: stage3/configs/transform/ops_fps_np2
```

`stage3/configs/transform/ops_fps_np2.yaml` 的核心变化：

```yaml
__target__: patch
patch_size: 1000
num_patches: 2
```

输出目录：

```text
stage3/experiments_hard005_tail015_bs16_fps_np2_from362/
```

后续使用：

```text
stage3/experiments_hard005_tail015_bs16_fps_np2_from362/stage3_checkpoint_hard005_tail015_bs16_fps_np2_from362_joint_461.pkl
```

### 6.7 Stage3: num_patches=4、bs8 续训到 544

这一阶段从 461 checkpoint 继续训练，将训练 patch 数改为 4，batch size 降为 8，并使用更稳的学习率组合。

```bash
python stage3/run.py \
  --task stage3/configs/task/joint_from_461_hard005_tail015_bs8_fps_np4_100
```

关键配置：

```yaml
init_joint_ckpt: ckp/stage3_checkpoint_hard005_tail015_bs16_fps_np2_from362_joint_461.pkl
start_epoch: 462
joint_epochs: 561
joint_batch_size: 8
stage2_lr: 0.00003
stage1_lr_mult: 0.05
stage1_loss_weight: 0.15
lambda_hard: 0.005
hard_tail_ratio: 0.15
components:
  transform: stage3/configs/transform/ops_fps_np4
```

`stage3/configs/transform/ops_fps_np4.yaml` 的核心变化：

```yaml
__target__: patch
patch_size: 1000
num_patches: 4
```

最终推理使用：

```text
stage3/experiments_hard005_tail015_bs8_fps_np4_from461/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl
```

如果不重新训练，可直接使用：

```text
ckp/stage3_checkpoint_hard005_tail015_bs8_fps_np4_from461_joint_544.pkl
```

## 7. 结果验证

如果需要验证新生成结果是否与已有提交结果一致，可以使用：

```text
stage3/compare_prediction_dirs.py
```

示例：

```bash
python stage3/compare_prediction_dirs.py \
  /path/to/reference_result_dir \
  stage3/final_runs/544_b_200/final \
  --test-list stage3/datalist/test.txt \
  --csv stage3/final_runs/compare_result.csv \
  --json stage3/final_runs/compare_result.json
```

该脚本会按相同相对路径匹配 `denoised.npy`，检查 shape 和 finite 值，并输出逐点 L2 距离、坐标绝对误差、RMSE 等指标。如果怀疑点顺序改变，可加：

```bash
--nearest-neighbor
```

本次复现与提交结果的历史比较结果为：

```text
samples: 200
point_l2_mean_mean_over_samples: 3.238e-6
point_l2_p95_mean_over_samples: 1.096e-5
point_l2_p99_mean_over_samples: 3.334e-5
point_l2_max_max_over_samples: 0.007598
```

整体逐点误差在 `1e-6` 到 `1e-5` 量级，可认为复现结果与提交结果高度一致。不同 CUDA/Jittor 缓存、GPU 调度或编译器下可能出现微小浮点差异，不能保证 zip 逐 bit 相同。

## 8. 重要文件速查

```text
stage1/run.py
  Stage1 训练入口，训练 PGD1。

stage2/run.py
  Stage2 训练入口，加载 Stage1 checkpoint，训练 PGD2。

stage3/run.py
  Stage3 联合训练入口，加载 Stage1/Stage2 或 joint checkpoint，联合训练 PGD1/PGD2。

stage3/final_inference.py
  最终推理入口，包含 PGD1/PGD2 前向、法向坐标系、local-alpha、Surface Gate、Coverage Repair 和打包。

stage3/src/model/ops.py
  KNN、PCA normal frame、patch_based_denoise、adaptive merge 等核心操作。

stage3/src/data/augment.py
  数据增强、patch 构造和 FPS 风格 patch center 采样。

stage*/configs/data/train.yaml
  训练数据根目录和 datalist 配置。换机器复现时优先检查这里。

stage*/configs/task/*.yaml
  每个训练/续训阶段的主配置，包含 checkpoint、训练轮数、学习率、loss 权重和组件路径。

ckp/
  本次复现链路保留的关键 checkpoint。
```

## 9. 常见问题

1. **Jittor 首次运行很慢**：这是 C++/CUDA 算子即时编译。建议先单进程运行一次，不要多个进程同时首次编译。
2. **找不到训练数据**：检查 `stage*/configs/data/train.yaml` 中的 `input_dataset_dir`，再检查 list 中每一行是否能拼到真实文件。
3. **找不到测试数据**：检查 `--input-root` 和 `--test-list`。每行必须能拼成 `<input-root>/<entry>/noisy.npy`。
4. **找不到 checkpoint**：检查 task YAML 中的 `load_ckpt`、`stage1_ckpt`、`stage2_ckpt`、`init_joint_ckpt`。换机器后推荐先指向 `ckp/` 中的相对路径。
5. **24GB 显存 OOM**：最终推理建议独占一张 4090；训练 `num_patches=2` 使用 bs16，`num_patches=4` 使用 bs8。不要为了省显存修改最终推理 patch 参数，否则不再是同一提交策略。
6. **压缩包目录不对**：最终 zip 内应直接是 `shapenet/<category>/<sample-id>/denoised.npy`，不要把 `final/` 或 `stage3/final_runs/` 这类外层目录打进去。
7. **不要做 `CD+4`**：历史本地分数校准不应写入预测点坐标，提交文件只保存模型生成的降噪点云。
