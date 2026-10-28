# 16/20自由度主轴最小项目

仅保留动力学、RMS参数标定、物理数据生成及完整参数配置，无绘图、报告生成或临时测试依赖。

## 目录

```text
src/main.py                 主线入口
src/generation.py           物理数据生成与导出
src/calibration.py          六个安装参数的RMS-only标定
src/spindle/                仅机械方程、安装链、Numba后端
utils/                      配置、读取、信号处理、评价
config/model16.json         后外圈绑定H的16自由度完整参数
config/model20.json         四外圈独立的20自由度RMS冻结完整参数
config/calibration_targets.json  真实健康fit/val参考RMS
tmp/                        全部任务临时脚本，不提交
docs/                       交接、计划、整理记录
data/                       原始和上游数据，只读、未清理
```

## 安装与运行

Python ≥3.10。现有环境：`D:/Softwares/miniconda3/envs/ZMSoftware/python.exe`。

```powershell
python -m pip install -e .
python -B -m main generate --config config/model16.json --output ../DataGeneration_work/model16_c1
python -B -m main generate --config config/model20.json --output ../DataGeneration_work/model20_c1
```

不安装时，从仓库根设置 `$env:PYTHONPATH='src;.'`，再运行同样命令。

默认480完整转、65536 Hz原生状态，导出8192 Hz三通道10–900 Hz速度。主要产物：`native.npz`（q/v/acc、接触力、测量位移和状态映射）、`signal.npz`（三通道、有效段掩码）、`per_revolution.csv/.npz`（逐圈标签）、实际配置、指标和稳态JSON。保留物理幅值，不施加β。

```powershell
# 20自由度改变前间隙时，必须显式指定两只后间隙：
python -B -m main generate --config config/model20.json --gaps 1 4 10 --rear-gaps 2 2 --output ../DataGeneration_work/rear_fixed_control
```

无`--gaps`时使用配置里的实际间隙，包括独立后间隙。配置的`clearances_um`列出1–10 μm等距25点，**不会自动启动扫描**。可用`--turns`/`--warmup`短时检查；短时可运行不等于稳态。退出码0表示求解完成且来源分窗RMS通过，2表示导出但RMS超差；求解异常保存失败信息并非零退出。

生成入口依次处理全部指定工况，逐级保存验收结果；稳态超差或未知状态不自动终止后续工况。全部处理后，存在未通过项时退出码为2，汇总状态为`COMPLETED_WITH_FAILURES`。每级独立静平衡、零初速冷启动；单次生成默认只做一次尝试。通用入口不执行逐级h/N认证，生成验证时是否继续由使用者决定。

按用户最新要求，唯一验收项是来源三通道10–900 Hz速度RMS的四窗六对比较；20自由度配置为5%，16自由度默认2%，绝对底限均为1e−4 mm/s。其他指标不设置通过/失败门槛，融合波形不做RMS稳态验收。原始位移、接触量及阶次幅值仍可保存。求解失败另存`partial_native.npz`（`PARTIAL_NOT_FORMAL`）、`failure.json`（求解诊断、源码CRC32、异常堆栈）；RHS抛异常只能保存最后确认的checkpoint。完整输出保留接受步长及绝对时刻。

## 配置与标定

两份模型配置保留全部机械质量、惯量、轴承几何/本构、配合间隙/刚阻尼、安装链、激励、求解容差、采样和测量参考，不依赖旧实验目录。`base_v8.parameters.model_dof`是基础机械12/16自由度，`structural_dof`才是完整16/20自由度。

实际安装矩阵以`installation.actual_link_stiffness_N_m`与`actual_link_damping_N_s_m`为准。冻结运行严格保留；标定入口改安装参数时同步重算实际K/C，避免参数改了而矩阵仍锁在旧值。

当前生产参数已经用户冻结；下面命令仅用于将来明确要求的新标定：

```powershell
python -B -m main calibrate --config config/model20.json --targets config/calibration_targets.json --output ../DataGeneration_work/new_calibration --max-evaluations 30
```

只拟合β前三通道10–900 Hz RMS，fit选参，val比值只报告，不设[0.5,2]等额外合格线。默认低精度搜索，再恢复配置中的正式数值设置确认选定候选；候选、失败和停止原因全部保留。求解完成且来源分窗RMS通过时输出`frozen_config.json`；预算耗尽不等于优化收敛。

标定参考是已有真实健康父片段→run中位数→split中位数；不含H_test。本最小项目不保留原始采集转换/筛选脚本，换数据时在`tmp/`完成准备，另保存来源和run划分，不能把测试数据混入目标。

## 交接

先读[AGENTS.md](AGENTS.md)。[交接文档](docs/HANDOVER.md)、[下一步实验计划](docs/EXPERIMENT_PLAN.md)、[整理及回归记录](docs/CLEANUP.md)。旧版内外圈点蚀模型可在 Git 历史提交 `8b4cc4b` 中查看；原始数据不随仓库提交。

2026-10-27新一轮模型校核、25级同步扫描、单背景合成和TCN结果见[实验报告索引](docs/EXPERIMENT_REPORT_20261027.md)。25级已导出，其中14级稳态、11级未稳态；条件合成与完整数值认证分开记录。

最新20级实验的[参数、主要结果、图和动画](docs/results/sync20_20261028/README.md)保存在仓库中；原始波形、逐窗数据、TCN权重和临时脚本未提交。完整本地数据位于 `../DataGeneration_work/`。
