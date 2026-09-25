# `upload.sh` 使用说明

这个脚本用于把随机噪声压制实验中已经训练完成的模型文件，批量整理并上传到 Hugging Face 模型仓库。

当前脚本上传的是每个实验目录下的两个文件：

- `checkpoints/best.pt`
- `config.yaml`

例如本地：

```text
/data/pcdisk0/data/results/random_noise/random_noise_unet_base_gaussian_snr0_seed42/checkpoints/best.pt
/data/pcdisk0/data/results/random_noise/random_noise_unet_base_gaussian_snr0_seed42/config.yaml
```

会被映射到 Hugging Face 仓库中的：

```text
models/unet/gaussian_snr0_seed42/best.pt
models/unet/gaussian_snr0_seed42/config.yaml
```

## 1. 脚本位置

```bash
scripts/random_noise_suppression/upload.sh
```

## 2. 上传前需要准备什么

在运行脚本前，需要先确保：

1. 当前机器已经安装 `git`
2. 当前机器可以访问 Hugging Face
3. 你对目标仓库有写权限
4. Hugging Face 认证已经配置好，或者后续 `git push` 时可以输入 access token

可以先检查：

```bash
git --version
```

## 3. 脚本里几个关键参数

`upload.sh` 顶部有几个常用配置：

### `HF_REPO_URL`

目标 Hugging Face 仓库地址，例如：

```bash
HF_REPO_URL="https://huggingface.co/SeisProcBench/SPBench-Model"
```

### `RESULTS_ROOT`

本地训练结果根目录，例如：

```bash
RESULTS_ROOT="/data/pcdisk0/data/results/random_noise"
```

### `MODEL_LIST`

控制上传模型的顺序，也方便只上传部分模型，例如：

```bash
MODEL_LIST=("unet" "dncnn" "res_unet" "atten_unet" "SCRN")
```

如果你只想先上传 `unet`，可以改成：

```bash
MODEL_LIST=("unet")
```

### `DRY_RUN`

是否只打印、不真正执行。

- `DRY_RUN=1`：只显示准备复制和提交什么，不真正执行
- `DRY_RUN=0`：真正执行复制、git add、git commit

### `PUSH`

是否在 commit 后自动推到远端。

- `PUSH=0`：只做本地 commit，不推远端
- `PUSH=1`：commit 后自动 `git push`

## 4. 推荐使用顺序

建议按下面顺序使用。

### 第一步：先 dry-run 检查映射

先设置：

```bash
DRY_RUN=1
PUSH=0
```

然后运行：

```bash
bash upload.sh
```

这一步不会真正上传，只会打印：

- 找到哪些实验目录
- 本地 `best.pt` / `config.yaml`
- 它们会被放到 Hugging Face 仓库里的什么位置

如果输出路径符合预期，再继续下一步。

### 第二步：正式复制并本地提交

把脚本改成：

```bash
DRY_RUN=0
PUSH=0
```

然后运行：

```bash
bash upload.sh
```

这一步会：

- clone 或复用本地 Hugging Face 仓库
- 复制文件到仓库目录
- `git add`
- `git commit`

但不会推送到远端。

运行后可以检查：

```bash
cd hf_random-noise-attenuation
git status
git log --oneline -n 5
```

如果看到本地已经生成新 commit，就说明这一步成功了。

### 第三步：推送到 Hugging Face

确认本地内容没问题后，把脚本改成：

```bash
DRY_RUN=0
PUSH=1
```

再运行：

```bash
bash upload.sh
```

如果之前已经生成过本地 commit，也可以直接进入本地 Hugging Face 仓库执行：

```bash
git push
```

## 5. 如何判断是否真的上传成功

### 本地检查

进入本地 Hugging Face 仓库后：

```bash
git status
git branch -vv
```

如果看到：

- `nothing to commit, working tree clean`
- 本地分支不再显示 `ahead of origin/main`

说明本地和远端已经同步。

### 远端检查

打开 Hugging Face 仓库网页，确认是否能看到类似目录：

```text
models/unet/gaussian_snr0_seed42/best.pt
models/unet/gaussian_snr0_seed42/config.yaml
```

## 6. 常见情况说明

### `Required command not found: git`

说明当前环境没有安装 `git`，需要先安装后再运行。

### `No changes to commit.`

这通常表示：

- 目标文件已经复制过了
- 本地仓库里已有相同内容
- 这次运行没有产生新的文件变化

不一定是错误，很多时候只是你重复运行了同一批上传。

### `Your branch is ahead of 'origin/main' by 1 commit`

说明：

- 本地 commit 已经成功
- 但还没有 push 到远端

这时直接执行：

```bash
git push
```

即可。

## 7. 当前脚本上传的内容范围

当前 `upload.sh` 只上传：

- `best.pt`
- `config.yaml`

不会上传：

- `README.md`
- 推理输出图
- `metrics_summary.json`
- `metrics_per_shot.csv`

如果后续你想把这些结果文件也整理上传，需要再扩展脚本逻辑。
