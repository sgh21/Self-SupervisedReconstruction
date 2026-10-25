# 主轴动力学建模与故障仿真

本项目保留主轴动力学模型、故障仿真、实测速度转换、同步位移和逐圈跳动计算。网络训练实现、模型和相关报告已移出工作目录，见[清理清单](docs/reports/cleanup_20261023.md)。

## 主要文件

| 文件或目录 | 作用 |
|---|---|
| `src/spindle/tai_bearing.py`、`tai_system.py` | V9准静态双接触局部求根、两侧套圈力/矩和12自由度装配 |
| `scripts/run_clearance_v9.py` | V9同激励机制对照、冻结U扫描及独立数值加密；保存真实失败状态 |
| `config/clearance_v9_config.json` | V9条件参数与旧U起始配置；实际验收另见V9试验目录 |
| `src/spindle/dynamics_model.py` | 6/7/8/10/12自由度分支；V8四轴承倾角接触力与局部矩、平衡点和时域积分 |
| `scripts/run_clearance_v8.py` | V8旧U机制对照、布局对照与冻结U扫描；实际认证另用`certify_clearance_v8` |
| `config/clearance_v8_config.json` | GMN 12自由度V8起始配置；正式冻结配置保存在V8试验目录 |
| `scripts/run_clearance_v3.py` | V3固定单只前轴承间隙；导出H速度/加速度、指定截面相对探头参考的逐转轨迹 |
| `config/clearance_v3_config.json` | V3统一物理与数值配置；明确标记assumed_demo |
| `scripts/run_clearance_sweep.py` | 固定前支承 S/T/H/O 八自由度配合间隙扫描、力学验证与联合收敛检查 |
| `config/clearance_config.json` | 配合间隙演示模型的一次分配、冻结参数及参数来源 |
| `config/platform_faults_20261022.json` | 当前平台模型的冻结参数与故障场景 |
| `scripts/generate_platform_faults.py` | 生成平台及各类故障响应，并检查数值精度 |
| `scripts/generate_spindle_sources.py` | 按独立参数组生成速度、原生轴/刀柄位移和逐圈标签 |
| `utils/runout_labels.py` | 从同步原生位移按完整转次计算X/Y最大值减最小值，单位µm |
| `scripts/prepare_velocity_data.py` | 原始加速度转为10–2048 Hz、8192 Hz采样的速度，单位mm/s |
| `data/runs/` | 原始采集数据，只读使用 |
| `data/derived_data/` | 已转换的实测速度及来源索引 |
| `experiments/spindle_sources_20261022/` | 已完成的156条原始仿真、312个速度窗及逐圈标签 |
| `experiments/spindle_healthy_sources_20261023/` | 另外32组健康物理仿真与检查 |
| `experiments/interface_sensitivity_20261023/` | HSK接口参数情景、敏感性和物理仿真历史结果 |
| `experiments/severity_sweep_20261021/` | 历史锥面面积损失扫描，保留原来的10–900 Hz处理版本 |
| `docs/reports/` | 模型审计、参数核对、故障响应及信号转换报告 |
| `docs/references/` | 文献原文、阅读笔记与核对证据 |

## 安装与运行

V9 保留12自由度和前后背靠背安装，分别求内外圈接触角、载荷、球离心力和陀螺力矩。壳体仍只有平动，轴向和外圈倾角约束反力单独保存。滚道曲率、固定位置预紧和结构阻尼是明确的条件假设；运行完成不等于数值验收或实机验证。V9证据保存在 `experiments/clearance_v9_tai_bearing/`，不会覆盖V8。

当前有限转速下的内圈失接触切换连续性未通过，原因是受载与脱离分支采用的球公转速度闭合不相容。已验收的整机结果限于始终保持内外双接触的条件域；失败状态和局部边界证据分别保留，不能把数值求根成功当作完整物理验证。

```powershell
$env:PYTHONPATH="$PWD/src"
python -B -m scripts.run_clearance_v9 --config config/clearance_v9_config.json --output experiments/clearance_v9_replay --group data_A --variant N3 --gaps 2 5 10 --turns 480 --warmup 3.5 --workers 2
python -B -m utils.clearance_v9_analysis --root experiments/clearance_v9_replay --group data_A --replay
```

上述命令固定旧U，不能替代健康标定、全部间隙扫描和独立加密。加密分别使用 `--step-factor 0.5`、`--root-factor 0.1` 或 `--N 1024`，每次写入新的 `--group`。采样加密由同一积分的65536/131072 Hz输出对照完成。原生数据、报告和文献受 `.gitignore` 排除，独立复核需另取数据包。

V8 在原12自由度上加入四只轴承的相对倾角接触，同一逐球接触关系同时输出径向力和局部力矩；外圈转动、轴向约束反力独立保存。入口为 `scripts.run_clearance_v8`，实际配置和证据在 `experiments/clearance_v8_tilt_contact/`。名义前后DB配对属于假设，另保留DF敏感性结果。模型边界见[参数合同](docs/reports/clearance_v8_parameter_contract.md)。

```powershell
$env:PYTHONPATH="$PWD/src"
python -B -m scripts.run_clearance_v8 --config experiments/clearance_v8_tilt_contact/parameters_v8_calibrated.json --output experiments/clearance_v8_replay --group data_B_recalibrated --gaps 2 3 5 7.5 10 --workers 3
# 已有原生数据的独立复算，不重新积分：
python -B -m utils.clearance_v6_analysis --root experiments/clearance_v8_tilt_contact --base-group data_B_recalibrated --integration-group convergence/h_half --angular-group convergence/n1024 --replay
```

实际运行命令（含高间隙480转、独立步长/接触离散细化和续接）见试验目录的 `运行命令.md`；上面第一条仅为基本扫描入口。最终数值状态须以分项验收为准。

V6 基线采用内部旋转质量约4 kg、两只前 GMN 6009 非线性接触和两个后 GMN 6007 健康作用点，仍为12自由度。先在共同前径向间隙2 μm下仅标定不平衡量U，使固定截面相对探头参考的逐转指标约2 μm；旋转力峰值限制为30 N。后续间隙扫描固定U，旧V3/V4/V5代码入口及结果保留。

```powershell
python -B -m utils.clearance_v6_parameters
python -B -m utils.clearance_v6_validation
python -B -m scripts.calibrate_clearance_v6 --output experiments/clearance_v6_replay
# 健康冷启动、步长/周向认证通过后，使用实际冻结配置扫描。
python -B -m scripts.run_clearance_v6 --config experiments/clearance_v6_GMN/parameters_v6_calibrated.json --output experiments/clearance_v6_gap_replay --gaps 0 2 3 5 7.5 10 --workers 3
```

参数来源见[合同](docs/reports/clearance_v6_parameter_contract.md)。实际标定历史、原生状态、独立数值细化及分项验收保存在`experiments/clearance_v6_GMN/`。健康目标属于条件标定，不能作为实机独立验证；主壳体振动频带10–900 Hz，辅助10–2048 Hz和900–2048 Hz，轴心指标来自未滤波原生位移。

从仓库根目录执行。现有环境为 `D:/Softwares/miniconda3/envs/ZMSoftware/python.exe`；依赖为NumPy、SciPy、h5py、Matplotlib。

```powershell
python -B -m pip install -e . --no-deps --no-build-isolation
python -B -m scripts.generate_spindle_sources --help

# 新仿真使用新的输出目录，不覆盖历史结果。
python -B -m scripts.generate_spindle_sources --config config/spindle_sources_20261022.json --output experiments/spindle_sources_replay --workers 4
python -B -m scripts.check_spindle_sources --sources experiments/spindle_sources_replay --output experiments/spindle_sources_replay_checks --plots

# 原始加速度转换；输出不得指向data/runs。
python -B -m scripts.prepare_velocity_data --raw-root data/runs --config config/velocity_data_20261022.json --all-runs --all-conditions --output data/derived_data/velocity_replay --workers 4
```

其余入口和作用见[完整清理清单](docs/reports/cleanup_20261023.md)。历史 `generate_severity` 和 `generate_outer_race_pitting` 入口还使用外部重建项目的旧2048 Hz健康参考，路径在 `config/severity_sweep.json`；新三通道生成入口不依赖该旧数据接口。

固定前支承配合间隙模型使用八自由度 `[Sx,Sy,Tx,Ty,Hx,Hy,Ox,Oy]`，与平台分支互斥。仅改变外圈与壳体的单侧径向间隙；输出 H 相对基座速度，以及 S 相对 H 的逐转 X/Y 位移峰峰值，保留1×。默认四个间隙为0、2、5、10 µm，0是理想对照，不是实测健康值。全部质量、支承与激励由配置冻结，扫描不重新拟合。

```powershell
# 复现主扫描、力学/圆轨迹检查、稳态与代表工况联合收敛检查，保存数据及图表。
python -B -m scripts.run_clearance_sweep --config config/clearance_config.json --output experiments/clearance_sweep_replay --validate
# 只运行一个间隙；不加--validate时numerical_pass保留为未完整验收。
python -B -m scripts.run_clearance_sweep --config config/clearance_config.json --output experiments/clearance_single_replay --clearance-um 5
```

参数来源及模型局限见[原文核对](docs/reports/clearance_sources_20261023.md)，实际数据和验收见[本轮报告](docs/reports/clearance_sweep_20261023.md)。每个NPZ同时保存高采样率原生状态、8192 Hz抗混叠后H速度、完整转次、接触诊断及参数元数据；两种采样率分别保存时间轴，不对幅值作归一化。

## V3 配合间隙模型

最新任务依据为 `GPTOrder/spindle_v3/`。V3保留S/T/H/O平动，新增两个转子倾角；目标前F1配合间隙是唯一扫描参数。健康前F2与后R1各承担原Kother的一半，位置、惯量、健康持续接触与固定探头支架均有明确演示假设。主观测为z=0.165 m刚转子截面相对固定探头支架，H固定方向速度/加速度由同次状态产生。旧8自由度代码和结果保留，主指标的参考不同，不能直接把V1和V3数值当同一量比较。

```powershell
python -B -m scripts.run_clearance_v3 --config config/clearance_v3_config.json --output experiments/clearance_v3_replay --gaps 0 0.79 1 --workers 3
```

[实际运行报告](experiments/clearance_v3_20261023/report.md)、[参数合同](docs/reports/clearance_v3_parameter_contract.md)、[力学说明](docs/reports/clearance_v3_mechanics.md)、[可行性预检](docs/reports/clearance_v3_feasibility.md)、[完整命令](experiments/clearance_v3_20261023/commands.md)。整个导出窗验稳，积分步长、周向方向数与原生采样分别核验；验收JSON分别判断数值、双输出和20 μm覆盖。运行成功不代表实机验证通过。

## 信号和模型边界

新仿真速度为三通道 `[N,3,8192]`、每窗1秒、mm/s；顺序为ACC2外壳X、ACC1外壳Y、ACC3平台Y。公共滤波截止频率为10和2048 Hz，两端约为−6 dB。原生位移另存，逐圈标签不由外壳速度积分得到。

机械方程未因本次清理改变。HSK-A40接口已由用户确认，但接触参数仍未标定；刀柄T是集中节点，没有完成15 mm测量截面的几何映射。借用的轴承参数、高速简化、平台传递差异仍需保留说明，见[模型逐式审计](docs/reports/spindle_model_formula_audit_20261022.md)。
