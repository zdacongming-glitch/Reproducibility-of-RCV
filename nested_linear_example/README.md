# Robust Cross Validation Simulation

这个项目用于复现实验性数值模拟：在一个二维线性回归例子中，比较普通交叉验证在鲁棒扰动下的模型选择表现，并生成原始结果、汇总表和图像。

核心比较对象是两个候选模型：

- `A1`: 只使用第一个特征 `x1`
- `A2`: 使用两个特征 `x1` 和 `x2`

数据生成模型为：

```text
y = beta1 * x1 + beta2 * x2 + eps
```

其中 `x1, x2` 为标准正态特征，`eps` 为均值为 0 的高斯噪声。代码会在不同样本量、训练/验证划分比例、扰动强度 `r` 和参数设置下，估计交叉验证选择模型是否等于真实鲁棒风险下的最优模型。

## 已实现功能

项目目前实现了三类 CV 准则：

- `adversarial`: 对验证残差使用点态最坏情形扰动分数。
- `aligned_adversarial`: 使用 aligned adversarial 结构，并通过 localized Gaussian-Hermite 绝对均值估计器估计交叉项。
- `stochastic`: 在验证阶段对 `x2` 加入均匀随机扰动。

每次仿真会记录：

- 每个候选模型的 CV 分数：`cv_a1`, `cv_a2`
- 每个候选模型的闭式鲁棒风险：`loss_a1`, `loss_a2`
- CV 选择模型和真实鲁棒最优模型
- 是否选中最优模型
- 有限样本切换阈值：`threshold_adv`, `threshold_sto`
- `delta_cv = cv_a2 - cv_a1`

## 实验设计

### Experiment 1

固定 `beta1=1.0`, `beta2=1.0`, `sigma=0.5`，考察不同训练比例和扰动半径下的模型选择行为。

默认设置：

- 样本量：`n = 1000, 2000`
- 训练比例：`0.1, 0.2, ..., 0.9`
- adversarial 半径：`0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2.0`
- stochastic 半径：包含 `sqrt(3)` 附近的一组密集取值

### Experiment 2

固定 `beta1=1.0`, `beta2=1.0`, `sigma=0.5`，考察样本量增大时 CV 选择是否趋于鲁棒最优模型。

默认设置：

- 样本量：`100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000`
- 训练比例：`0.1, 0.2, ..., 0.9`
- adversarial 半径：`0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2`
- stochastic 半径：`0.5, 0.8, 1.0, 1.5, sqrt(3), 2.0, 2.2`

## 项目结构

```text
robust_cv_sim/
  simulation.py   # 数据生成、模型拟合、CV 准则、闭式风险、实验行生成
  reporting.py    # raw 结果写入、汇总表生成、绘图
  runner.py       # 分块运行入口，支持 resume 和进度条
  cli.py          # 简单非分块运行入口
tests/
  test_simulation.py
  test_runner.py
outputs/          # 默认实验输出目录
```

## 安装依赖

建议使用 Python 3.11 或更高版本。

```bash
python -m pip install -e .
python -m pip install numpy pandas matplotlib seaborn rich pytest
```

当前 `pyproject.toml` 只声明了包元信息和 pytest 配置，运行依赖需要按上面命令安装。

## 运行实验

推荐使用分块 runner。它支持进度条、断点续跑和 chunk 合并，适合完整实验。

运行全部实验：

```bash
python -m robust_cv_sim.runner
```

只运行 experiment 2：

```bash
python -m robust_cv_sim.runner --experiments experiment_2
```

减少重复次数用于快速 smoke test：

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --reps 10 --output-dir smoke_outputs
```

跳过绘图，只生成 raw 和 summary：

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --skip-plots
```

默认参数：

- `--reps 1000`
- `--seed 20260319`
- `--output-dir outputs`
- 默认开启 `--resume`

如果希望忽略已有 chunk 并重新运行：

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --no-resume
```

## 输出文件

每个实验会在输出目录下生成独立子目录，例如：

```text
outputs/
  experiment_2/
    experiment_2_raw.csv.gz
    experiment_2_summary.csv
    experiment_2_threshold_summary.csv
    plots/
      exp2_adversarial_trainratio_0.1_consistency.png
      exp2_adversarial_trainratio_0.1_delta_cv_boxplots.png
      ...
```

主要文件说明：

- `*_raw.csv.gz`: 每次重复、每个配置和每个 `r` 的完整结果。
- `*_summary.csv`: 按实验配置聚合后的均值、选择概率和置信区间。
- `*_threshold_summary.csv`: 有限样本切换阈值的分布汇总。
- `plots/`: 自动生成的可视化结果。

Experiment 2 的图包括：

- consistency 图：横轴为 `n`，纵轴为 `P(select robust-optimal)`，不同曲线对应不同 `r`。
- `delta_cv` boxplot：展示不同样本量和扰动强度下 `cv_a2 - cv_a1` 的分布。

## 在代码中调用

可以直接使用 Python API 运行小规模实验：

```python
from robust_cv_sim import SimulationConfig, run_experiment_2

config = SimulationConfig(
    reps=10,
    split_grid=(0.5,),
    exp2_n_grid=(100, 500, 1000),
)

df = run_experiment_2(config)
print(df.head())
```

也可以覆盖 experiment 2 的扰动半径：

```python
config = SimulationConfig(
    exp2_radius_grid_adv=(0.5, 0.7, 0.8, 1.0),
)
```

## 测试

运行全部测试：

```bash
python -m pytest
```

当前测试覆盖了：

- adversarial 和 stochastic CV 公式
- Gaussian-Hermite 估计器的基本数值性质
- 闭式鲁棒风险公式
- 两个实验的 chunk 枚举
- chunk 写入、合并和 resume 相关逻辑
- summary 和绘图辅助函数
- experiment 2 默认 adversarial 半径集合
