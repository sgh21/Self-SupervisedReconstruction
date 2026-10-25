# TCN 自监督重建

用 7000 rpm 稳态振动速度建立正常重建基线。输入和输出均为三通道、
1 秒、2048 Hz 波形 `[B, 3, 2048]`，通道顺序为 ACC1、ACC2、ACC3，单位 mm/s。
现有数据缺少真实故障标签，重建误差不能直接解释为故障类型或回转精度。

## 目录

```text
config/       # 日常训练配置、两组抗干扰测试配置
model/        # 自定义门控 TCN、标准 TCN 及模型构建
src/          # 数据准备和训练入口
test/         # 重建测试、抗干扰测试入口
utils/        # 数据读取、归一化、指标、增强和可视化
data/         # 原始数据、派生数据、正式训练数据集
experiments/  # 正式权重、选型依据、配置快照和实验结果
reports/      # 历史研究报告
AGENTS.md     # 工作规则和研究结论
```

`data/runs/` 只读。训练、验证和测试按完整 run 隔离；窗口不是独立机械试验。

## 环境与训练

在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate spindle-reconstruction
python -m src.train --config config/tcn.yaml
tensorboard --logdir experiments/tcn
```

已有环境时直接激活。默认使用 CUDA，CPU 运行修改 `training.device: cpu`。
配置中的数据、权重和输出相对路径均以项目根目录为基准。

日常训练只维护 `config/tcn.yaml`。`model.implementation` 设为 `custom` 或 `standard`
即可切换网络；两种实现分别位于 `model/tcn.py` 和 `model/tcn_standard.py`。
当前配置为 32 通道、5 个 TCN block，编码后形状为 `[B, 32, 128]`；
训练数据为 7 月 22—24、26 日，验证为 7 月 25 日，`test_runs` 为空。
这些 run 已参与历史开发，不能称全新盲测。

每次训练在 `experiments/tcn/` 下创建独立目录，保存 `best.pt`、`config.yaml`、
`splits.json`、指标、训练摘要和 TensorBoard。归一化只用训练集拟合，随权重保存；
普通训练按验证归一化 MSE 选模。配置含 `robustness_validation` 时，按其中的干净/受扰
验证加权 MSE 选模。测试不参与选择。

`visualization` 控制固定验证案例的导出频率，训练目录中的 `visualizations/index.html`
可查看三通道波形、频谱和局部放大图；TensorBoard 的 Images 页面也会显示案例。

## 数据准备

默认直接读取已有的 `data/organized_data/reconstruction_velocity_temperature/7000rpm`。
需要重新生成时，先把配置的 `data.root` 改成新的输出目录，再运行：

```powershell
python -m src.prepare_data --config config/tcn.yaml
```

入口按传感器映射读取完整 10 秒段，筛选平台开始 600 秒后、`7000 ± 10 rpm` 且
`speed_ok=true` 的稳态数据，将加速度转为 10–900 Hz、2048 Hz 速度，再切为 1 秒窗口。
HDF5 保存 run、原始段号和时间信息；不覆盖已有 HDF5。

## 测试

测试入口统一放在 `test/`，从项目根目录用 `python -m` 运行。
重建测试读取配置的 `data.test_runs`，并检查它们与 checkpoint 的训练/验证 run 不重叠。
日常训练配置没有测试集；复测已有实验时使用该实验保存的配置。例如：

```powershell
python -m test.test experiments/robustness_date_split_20261014/formal/custom/clean/20261014-194756-154457/best.pt --config experiments/robustness_date_split_20261014/formal/custom/clean/20261014-194756-154457/config.yaml
```

默认打印指标；加 `--output <新JSON路径>` 可保存，已有文件不覆盖。
网络结构和归一化读取 checkpoint，输出时域及 Hann 幅值谱的重建指标。

两份抗干扰配置对应不同数据和权重，分别保留：

```powershell
# 7 月 26 日留出 run：本轮隔离，但历史已使用
python -m test.test_robustness --config config/robustness_date_test.yaml

# 8 月 23 日两个跨日期 run
python -m test.test_robustness --config config/robustness_test.yaml
```

测试会新建结果目录，保存配置、逐 run 指标、汇总和案例图。
人工干扰测试只评价这些条件下的重建能力，不代表真实故障检测有效性。

## 历史结果

- [按日期隔离的抗干扰实验](experiments/robustness_date_split_20261014/report.md)
- [跨日期抗干扰实验](experiments/robustness_20261014/report.md)
- [1 秒网络结构比较](reports/003_network_architecture_1s_comparison/report.md)
- [TCN 特殊工况验证](reports/004_tcn_1s_special_condition_validation/report.md)

历史训练参数以各正式运行目录的 `config.yaml` 为准，不在 `config/` 重复维护。
重新训练可将该快照直接传给 `src.train --config`；新权重生成后需同步测试配置的路径。
历史报告记录当时的实现与条件，部分旧模型和分析入口已不在当前精简代码中。
