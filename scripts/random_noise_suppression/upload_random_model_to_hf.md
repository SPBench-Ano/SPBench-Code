# 随机噪声压制模型上传说明

这份说明文档用于介绍随机噪声压制模型上传到 Hugging Face 的相关文件、它们之间的关系，以及日常运行方式。

当前涉及 3 个核心文件：

- `README_models.md`
- `upload_random_model_to_hf.py`
- `upload_random_model_to_hf.sh`

## 1. 三个文件分别是干什么的

### `README_models.md`

这个文件是 **Hugging Face model card 的静态模板**。

它主要放：

- 任务介绍
- 数据集说明
- 模型说明
- 训练/推理流程说明
- 固定的 benchmark 背景信息

这个文件本身**不直接上传为最终 README**，而是作为模板正文使用。

上传时，脚本会：

1. 先读取 `README_models.md`
2. 再自动扫描当前结果目录中的测试聚合结果
3. 自动生成结果表格
4. 把模板正文和自动表格拼成最终上传到 Hugging Face 的 `README.md`

所以可以把它理解为：

- **静态说明模板**

---

### `upload_random_model_to_hf.py`

这个文件是 **真正执行上传逻辑的 Python 脚本**。

它主要负责：

1. 扫描本地随机噪声压制结果目录
2. 找到每个实验目录中的：
   - `checkpoints/best.pt`
   - `config.yaml`
3. 按照固定规则映射到 Hugging Face 仓库路径，例如：

```text
random_noise_unet_base_gaussian_snr0_seed42/checkpoints/best.pt
-> models/unet/gaussian_snr0_seed42/best.pt
```

4. 调用 `huggingface_hub` 的 API 上传文件
5. 读取 `README_models.md`
6. 扫描 `*_seed_stats/metrics_summary_mean_std.json`
7. 自动生成并更新 Hugging Face 的 `README.md`
8. 如果远端已经有旧表格，会自动合并已有结果和新结果

所以可以把它理解为：

- **上传执行器**
- **README 自动生成器**

---

### `upload_random_model_to_hf.sh`

这个文件是 **日常给组员直接运行的 shell 入口**。

它的作用是把常用配置集中起来，避免每次手动敲很长的 Python 命令。

常改的内容基本都放在这个 `.sh` 文件里，比如：

- Hugging Face 命名空间
- 仓库名
- 结果目录
- 要上传哪些模型
- 模型显示名
- 是否 dry-run
- 是否更新 model card

所以可以把它理解为：

- **上传控制面板**
- **给日常使用者改参数的入口**

通常情况下，组员只需要改这个 `.sh` 文件，不需要直接改 `.py`。

## 2. 三者之间的关系

三者的调用关系如下：

```text
upload_random_model_to_hf.sh
        ↓
upload_random_model_to_hf.py
        ↓
读取 README_models.md
        ↓
扫描结果目录并生成最终 README.md
        ↓
上传 best.pt / config.yaml / README.md 到 Hugging Face
```

简单说就是：

- `.sh`：负责“怎么运行”
- `.py`：负责“具体怎么上传”
- `README_models.md`：负责“README 的静态正文模板”

## 3. 日常上传时推荐怎么做

### 第一步：设置 Hugging Face 环境变量

在 Linux 终端里先设置：

```bash
export HF_NAMESPACE=SPBench
export HF_TOKEN=你的_huggingface_token
```

其中：

- `HF_NAMESPACE`：Hugging Face 用户名或组织名
- `HF_TOKEN`：有写权限的 token

---

### 第二步：修改 `upload_random_model_to_hf.sh`

常用配置在文件顶部。

例如：

```bash
REPO_NAME="random-noise-attenuation"
RESULTS_DIR="/data/pcdisk0/data/results/random_noise"
MODEL_LIST=("unet")
DRY_RUN=1
NO_MODEL_CARD=0
```

#### `MODEL_LIST`

控制这次上传哪些模型，以及上传顺序。

例如只上传 `unet`：

```bash
MODEL_LIST=("unet")
```

例如上传多个模型：

```bash
MODEL_LIST=("unet" "res_unet" "atten_unet")
```

#### `MODEL_DISPLAY_LIST`

控制 Hugging Face README 结果表格中的显示名称。

例如：

```bash
MODEL_DISPLAY_LIST=(
  "unet=UNet"
  "dncnn=DnCNN"
  "res_unet=ResUNet"
  "atten_unet=Attention UNet"
  "SCRN=SCRN"
)
```

如果后续新增模型 `resnet`，可以这样加：

```bash
"resnet=ResNet"
```

#### `DRY_RUN`

- `1`：只打印要上传什么，不真正上传
- `0`：正式上传

#### `NO_MODEL_CARD`

- `0`：上传模型文件，同时更新 Hugging Face 的 `README.md`
- `1`：只上传模型文件，不更新 model card

## 4. 推荐运行流程

### 先 dry-run 检查

第一次建议先设置：

```bash
DRY_RUN=1
```

然后运行：

```bash
bash scripts/random_noise_suppression/upload_random_model_to_hf.sh
```

这一步会打印：

- 目标仓库
- 选中的模型
- 哪些文件会上传
- `README.md` 是否会更新

如果这些输出没问题，再正式上传。

---

### 正式上传

把：

```bash
DRY_RUN=0
```

然后运行：

```bash
bash scripts/random_noise_suppression/upload_random_model_to_hf.sh
```

这一步会真正上传：

- `best.pt`
- `config.yaml`
- 自动生成后的 `README.md`

## 5. 当前上传脚本会上传哪些文件

当前固定上传：

- `checkpoints/best.pt`
- `config.yaml`

并且会自动更新：

- Hugging Face 仓库中的 `README.md`

不会上传：

- `metrics_per_shot.csv`
- `metrics_summary.json`
- 可视化图片
- `.npy` 文件

## 6. README 表格是怎么自动更新的

当前 `upload_random_model_to_hf.py` 会扫描：

```text
*_seed_stats/metrics_summary_mean_std.json
```

这些是多随机种子聚合后的 mean/std 结果文件。

脚本会：

1. 读取这些 JSON
2. 按噪声类型和 SNR 组织结果
3. 自动生成 markdown 表格
4. 与远端已有结果合并
5. 更新 Hugging Face 仓库里的 `README.md`

这意味着：

- 如果这次只上传 `unet`
- 那么表格会新增或更新 `unet` 的结果
- 原来已经上传过的其他模型结果不会丢失

## 7. 常见问题

### 如果 Hugging Face 上已经有同名文件，现在是覆盖还是跳过？

当前逻辑分两类：

#### 1. 模型文件：默认跳过，不覆盖

对于这些文件：

- `best.pt`
- `config.yaml`

脚本会先检查 Hugging Face 仓库里是否已经存在相同路径。

如果远端已经有，例如：

```text
models/unet/gaussian_snr0_seed42/best.pt
```

那么当前脚本会：

- 直接跳过
- 不覆盖旧文件

也就是说，当前策略是：

- **已存在的模型文件默认保留**
- **避免误覆盖**

#### 2. `README.md`：会重新上传更新

`README.md` 不走“存在即跳过”的逻辑，而是每次上传时都会重新生成并上传。

但这里不是简单粗暴覆盖，而是：

1. 先读取远端已有的 `README.md`
2. 提取其中已有的自动结果数据
3. 合并当前本地新结果
4. 再生成新的完整 `README.md` 上传

所以它的行为更准确地说是：

- **README 会更新**
- **结果表是合并式更新**

#### 总结

当前默认策略是：

- `best.pt` / `config.yaml`：**已存在就跳过**
- `README.md`：**会更新**

如果以后需要“同一路径强制覆盖模型文件”，需要再单独扩展脚本逻辑。

### 为什么模型上传了，但 README 看起来没变化？

可能原因：

1. 远端页面缓存，刷新后才显示
2. 当前结果和已有结果完全一样
3. 对应的 `*_seed_stats/metrics_summary_mean_std.json` 没生成

---

### 为什么要保留 `README_models.md`，不直接把完整 README 写死？

因为静态说明和动态结果是两类内容：

- 静态说明：任务、数据、模型、流程
- 动态结果：不同模型、不同噪声类型、不同 SNR 的测试表格

把这两部分拆开，更方便长期维护。

---

### 新增模型时要改哪里？

通常只需要改 `upload_random_model_to_hf.sh`：

1. 在 `MODEL_LIST` 里加模型名
2. 在 `MODEL_DISPLAY_LIST` 里加显示名

只要实验目录命名符合当前规则，例如：

```text
random_noise_resnet_base_gaussian_snr0_seed42
```

Python 上传脚本通常不用改。

## 8. 一句话总结

日常使用时可以这样理解：

- **改模板说明**：改 `README_models.md`
- **改上传逻辑**：改 `upload_random_model_to_hf.py`
- **改运行参数 / 选模型 / dry-run / 是否更新 README**：改 `upload_random_model_to_hf.sh`

对大多数组员来说，平时只需要改并运行：

```bash
scripts/random_noise_suppression/upload_random_model_to_hf.sh
```

就够了。
