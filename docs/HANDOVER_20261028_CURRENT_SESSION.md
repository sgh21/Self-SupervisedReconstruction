> **2026-10-28最新留存整理：** 用户要求仅留存最后一轮及其复现依赖；删除旧实验与临时脚本的操作被自动审批阻止，当前尚未清理，清理清单已保存。请先查看 [最终留存与复现说明](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/README_留存与复现.md) 和 [历史记录](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/历史记录.md)。

# 当前会话交接｜2026-10-28

> **最新用户规则覆盖下方历史验收口径：以后只保留来源仿真的分窗RMS判定，其他限值、观测质量及融合RMS检查均取消。** 当前20级按此口径见[更新后的结果](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/rms_only/report.md)。原历史输出与旧脚本仍保留，复现旧过程须使用其输出目录内的源码快照。

> 后续会话已完成[同步20级联合实验](SYNC20_JOINT_RMS_STABILITY_REPORT_20261028.md)：最终独立配置c07，四隙同步0–5 μm共20级，来源RMS19/20、物理15/20，融合RMS15/20，旧TCN80窗完成。根model20未覆盖。请先读该验收索引；下文保留本文件原会话的固定后3.5 μm历史交接，不能作为新一轮参数或进度。

> 用户已确认该后续实验不更新AGENTS.md，保留`tmp/sync20_joint_rms_stability_20261028/`供复现，不自动清理或移入主线。

本文件记录本会话的目标演变、关键改动、实际结果和接续边界。截至本次交接，最新扫描、β更新、合成及旧TCN验证均已完成，没有待续的动力学生成或训练任务。本文不授权追加试验。

## 1. 接手先看这几件事

- 项目：`D:/WorkSpace/ZMProject/DataGeneration`；TCN工程：`D:/WorkSpace/ZMProject/Self-SupervisedReconstruction`。
- 16/20自由度模型都保留。此前“回退16自由度”是某轮数据任务选择，并未删除20自由度模型。
- **最新通过扫描使用“原20自由度实际刚度K＋16自由度实际阻尼C”，不是根目录当前c24参数，也不是完整16自由度K/C。**
- 最新扫描：后R1=R2固定3.5 μm，前F1=F2从0到10 μm、步长0.5 μm，共21组。21组全部通过本次480转物理/RMS分窗检查。
- 最新下游：复用上述数据，重算β、合成21条信号，复用之前健康数据训练的TCN验证84个1秒窗口；**没有重新生成仿真，也没有重训TCN**。
- 最新用户要求结论只写验收报告，不更新AGENTS.md。扫描临时脚本已明确要求保留；最新下游脚本的保留问题已询问，尚未收到该项答复，暂保留，勿自行删除。

仓库AGENTS、旧HANDOVER和EXPERIMENT_PLAN中的历史25级同步扫描、首失败即停等段落，不能代替本会话最新指令。接手仍应先读根目录AGENTS.md，再读README和交接；新任务以用户最新要求为准。

## 2. 目标与关键代码改动

最初目标是修复20自由度模型的工程问题，并在不随意改变物理参数、激励或验收门槛的前提下定位非稳态。随后用户分别授权了后间隙对照、16自由度回退、阻尼对齐、局部RMS搜索、固定参数扫描，以及β/合成/TCN验证。

| 项目 | 当前实现与边界 |
|---|---|
| 相干阶次估计 | Hann加权联合SVD最小二乘；不能独立证明可靠性的阶次保留UNKNOWN/INCONCLUSIVE，不将“未知”解释为物理非稳态 |
| RMS稳态门限 | 20自由度速度RMS分窗相对门限为5%，绝对底限1e−4 mm/s；其他物理指标仍使用原门限，h/N数值对照不随之放宽 |
| 后间隙默认联动 | 20自由度改变前间隙时，必须显式给出R1/R2；两只后间隙不再暗中跟随前端 |
| 积分失败保存 | 保存partial_native.npz、失败诊断、源码CRC32与异常信息；RHS异常只能保存最后确认checkpoint，不伪造失败试算状态 |
| 首失败停止 | 用户澄清这是当时生成验证的操作要求；现行通用入口处理全部指定工况并保留失败，最终存在失败返回2。已去掉代码层面的质量失败即停；正常执行异常仍保存错误并报出 |

主要代码位置：[生成入口](D:/WorkSpace/ZMProject/DataGeneration/src/main.py)、[生成与分窗检查](D:/WorkSpace/ZMProject/DataGeneration/src/generation.py)、[配置及显式间隙操作](D:/WorkSpace/ZMProject/DataGeneration/utils/configuration.py)、[指标](D:/WorkSpace/ZMProject/DataGeneration/utils/metrics.py)、[动力学目录](D:/WorkSpace/ZMProject/DataGeneration/src/spindle)。修复及回归证据见[R2验收报告](D:/WorkSpace/ZMProject/DataGeneration/docs/REPAIR_ACCEPTANCE_20261027_R2.md)。

## 3. 参数区别：不要再混用

实际安装矩阵优先于名义阻尼比ζ。仅改变ζ、却仍使用保存的实际C矩阵，不会实现预期阻尼变更。

最新稳定方案的实际安装参数如下，HC、CP的X/Y相同：

| 连接 | 实际K（N/m） | 实际C（N·s/m） |
|---|---:|---:|
| H-C，X/Y | 677963598.7080423 | 5713.423394700079 |
| C-P，X/Y | 274117277.0532099 | 13926.545316512464 |
| P-G，X | 7976840.755194302 | 6143.6017423138655 |
| P-G，Y | 35096312.58284531 | 29747.136011644598 |

有效完整输入：[最新扫描base_config.json](D:/WorkSpace/ZMProject/DataGeneration_work/rear_fixed3p5_frontscan_20261028/base_config.json)。参数ID保留为 `rear20_actual_C_from16_F3p5_R3p5_20261027`，名称里的3.5不代表每个工况都取3.5，实际四间隙以各case的actual_config.json为准。

根目录[model20.json](D:/WorkSpace/ZMProject/DataGeneration/config/model20.json)仍是此前用户冻结的 `rear20_local_rms_c24_userfrozen_20261027`，未被最新对照覆盖。其HC/CP/PGx/PGy刚度分别约为7.4576e8、2.71271e8、8.77452e6、3.86059e7 N/m，与上表不同。**继续使用最新稳定方案时，必须明确加载上述base_config，不能直接拿根配置当作同参数复现。**

20自由度仍保留OR1/OR2独立XY和后配合；H质量40.289190476 kg，两只后外圈各0.061904762 kg，总运动质量474.386 kg。不能把16自由度H质量直接覆盖进去，造成质量重复计算。仿真通道顺序为ACC2/Hx、ACC1/Hy、ACC3/Py，20自由度索引[4,5,19]；TCN顺序为ACC1/Hy、ACC2/Hx、ACC3/Py，推理前已换序。

## 4. 本会话各阶段的实际进展

| 阶段 | 已完成结果 | 证据 |
|---|---|---|
| 修复后，后间隙固定2 μm | 仅前1 μm首级；物理稳态通过，阶次可靠性未知导致当时按操作要求停止；后24级未运行。该级h/2、N1024、7秒热身及960转诊断比较通过 | [R2报告](D:/WorkSpace/ZMProject/DataGeneration/docs/REPAIR_ACCEPTANCE_20261027_R2.md) |
| 后间隙改1 μm，再回退16自由度 | 20自由度完成7级，前3.25 μm处物理失败后停止；16自由度确有35条仿真和35条融合波形，35/35物理稳态及逐级h/2、N1024、采样复核通过，1/3.75/10另有960转。健康RMS fit/val未全面达到[0.5,2]，阶次仍UNKNOWN；β16/21，健康TCN训练100轮 | [回退验收](D:/WorkSpace/ZMProject/DataGeneration/docs/REAR1_FALLBACK_ACCEPTANCE_20261027.md) |
| 固定前3.5，扫描后间隙 | 当时计划0–10 μm共25级；用户叫停，只完成20级至后7.916667 μm，物理7/20、RMS9/20，后5级未运行；未做h/N或长时认证 | [当时报告](D:/WorkSpace/ZMProject/DataGeneration_work/rear_gap_scan_front3p5_20261027/report.md) |
| 20自由度仅将C对齐16 | 前后3.5 μm通过；四窗RMS最大差约0.000406%，原20刚度保留。图中稳定结果真实有效 | [旧稳定单点](D:/WorkSpace/ZMProject/DataGeneration_work/rear20_damping16_F3p5_R3p5_20261027/report.md) |
| 局部RMS搜索并冻结c24 | 用户授权小范围搜索，选到边界后要求停止并冻结；搜索使用短片段，不能当成全扫描或完整数值认证 | [搜索输出](D:/WorkSpace/ZMProject/DataGeneration_work/local_rms_search20_20261027) |
| c24四间隙同步0–10扫描 | 21/21生成，物理18/21、RMS19/21；物理失败3.5、4.5、5 μm。完成当时的新β、单背景融合和健康TCN100轮 | [c24报告](D:/WorkSpace/ZMProject/DataGeneration/docs/SYNC21_C24_REPORT_20261027.md) |
| 将安装K/C都对齐16后复算 | 仅重算上述3点，全部物理/RMS未通过；最大RMS差6.255%、6.014%、15.293%。这不是旧“只对齐C”配置 | [完整K/C对齐报告](D:/WorkSpace/ZMProject/DataGeneration/docs/STRUCTURE16_ALIGNMENT_RECHECK_20261028.md) |
| 恢复原20K＋16C，重算同步3点 | 3.5 μm通过且完整速度波形与旧结果逐点相同；4.5 μm整体RMS通过，但Py非整数带5.153%略超5%；5 μm失败，Hx整体RMS差13.182% | [恢复参数复算](D:/WorkSpace/ZMProject/DataGeneration_work/rear20_oldK_C16_recheck_20261028/report.md) |
| 补充前5、后3.5单点 | 全部物理/RMS通过；三通道最大差约[3.5703e−5,5.6049e−5,1.5700e−5]%，E_dir=9.670863 μm | [单点报告](D:/WorkSpace/ZMProject/DataGeneration_work/rear20_oldK_C16_F5_R3p5_20261028/report.md) |
| 最新固定后3.5、扫描前0–10 | 21/21全部生成并通过物理/RMS检查；按用户要求每完成5组汇报曲线，5/10/15/20级快照均保留 | [最新扫描报告](D:/WorkSpace/ZMProject/DataGeneration_work/rear_fixed3p5_frontscan_20261028/report.md) |

这些结果说明不同K/C及间隙组合会改变响应，不能据此把历史所有非稳态单独归因于阻尼、后间隙或20自由度拓扑。

## 5. 最新21级扫描：可直接复用的数据

结果根目录：`D:/WorkSpace/ZMProject/DataGeneration_work/rear_fixed3p5_frontscan_20261028`。

- 每级独立静平衡、t=0、零初速冷启动；热身3.5秒，正式480转，原生65536 Hz、输出8192 Hz，N512，原积分容差和步长保持。
- 最大RMS分窗差为0.000419356%，位于前2.5 μm的Hx通道，远低于5%。模型E_dir范围0.378001–20.468995 μm，保留1×；不是实机几何精度证据。
- `cases/front_*um/`保留native.npz、signal.npz、actual_config.json、stationarity.json及逐圈指标；[数值总表](D:/WorkSpace/ZMProject/DataGeneration_work/rear_fixed3p5_frontscan_20261028/scan_summary.csv)、[累计曲线](D:/WorkSpace/ZMProject/DataGeneration_work/rear_fixed3p5_frontscan_20261028/figures/final_summary.png)、[独立核对](D:/WorkSpace/ZMProject/DataGeneration_work/rear_fixed3p5_frontscan_20261028/audits/audit_21.json)可直接查看。
- 临时代码在`tmp/rear_fixed3p5_frontscan_20261028/`，用户已明确保留。主线模型及根配置未改。21点分窗通过不等于21点h/N及长时数值认证。

## 6. 最新β、合成及旧TCN验证

结果根目录：`D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028`。主报告：[验收报告与全部图片](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/report.md)。

### 健康参考与β

复用已有同K/C的[四隙均1 μm健康仿真](D:/WorkSpace/ZMProject/DataGeneration_work/rear20_damping16_all1um_20261027/case/actual_config.json)。它与21级扫描逐字段核对过，只有间隙、相应标签、派生平衡初值和运行统计不同，没有新生成动力学。

β只用10个训练健康run的30个父片段拟合，冻结后检查3个验证run的9个父片段；测试背景未进入拟合。新β ID为 `beta_rear20_C16_all1um_fit10_r35_20261028`，启用18/21项。Q03整体中位相对误差val约4.35%、test约3.90%，但逐分量val失败ACC2 1×、ACC3 1×；test失败ACC2 1×、ACC1 5×、ACC3 1×，全部保留，未据验证曲线调β。

[β参数](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/beta.json)、[β验证](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/beta_validation.json)。1e−4 mm/s只是固定检测底限；数值误差预算未知保留null，不冒充置信界。

### 合成

固定测试背景`run_20260724_104047_middle_480turn`，沿用1–7×有符号幅值差编辑：保留实测相位和残差，负目标幅值裁零，不逐窗缩放。21条融合波形均已保存。

真正四隙1 μm参考的零注入恒等检查通过；**本扫描前1、后3.5 μm并非健康四隙1 μm，不能强制该点零注入**。该点有18项非零注入。

来源仿真物理稳态21/21通过；融合波形另做RMS检查，18/21通过，前0、0.5、1 μm未通过，最大通道分窗差分别约36.16%、8.57%、21.58%。这三点仍保留推理，在曲线中用橙色空心三角标记，不能把来源稳态替代融合稳态。

[合成数据清单](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/final_m1/manifest.csv)、[合成汇总](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/final_m1/summary.json)、[健康恒等检查](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/health_reference_identity.json)。

### TCN及指标

直接复用旧健康TCN，没有重训。输入2048 Hz、三通道、每窗2048点，沿用checkpoint的训练集Z-score；完整run train/validation/test为25/3/14。背景run属于test。输入/输出形状、通道、归一化、源码及精度兼容性已核对，21级共84窗完成推理。

权重：[best.pt](D:/WorkSpace/ZMProject/DataGeneration_work/sync21_c24_20261027/tcn/training/20261027-230351-862332/best.pt)，epoch100，22979参数，本轮CRC32=`2cab390f`。旧健康基准当时未保存权重CRC，只能追溯同一路径与处理口径；本次物理MSE和余弦不使用该健康基准作分母。

MSE为物理空间`mean((输入−重建)^2)`，单位(mm/s)²；余弦不去均值，每窗每通道计算点积除以L2范数乘积，再对三个通道和四窗等权平均，不是Pearson相关。

| 前间隙 μm | 模型E_dir μm | MSE (mm/s)² | 平均余弦 | 融合RMS |
|---:|---:|---:|---:|---|
| 1 | 2.316330 | 0.0000271744 | 0.989687 | 未通过 |
| 3.5 | 8.675356 | 0.609986 | 0.495272 | 通过 |
| 5 | 9.670863 | 0.0932395 | 0.542303 | 通过 |
| 10 | 20.468995 | 0.0197375 | 0.683214 | 通过 |

21级曲线明显非单调：MSE最大、余弦最低都位于前3.5 μm。E_dir与MSE的描述性Spearman约0.469，不能解释为实机精度预测；单背景、β逐分量失败及幅值外推均限制结论。

交付文件：[逐级指标](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/tcn_validation/levels.csv)、[逐窗指标](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/tcn_validation/windows.csv)、[输入与重建数组](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/tcn_validation/reconstructions.npz)、[指标曲线](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/figures/metrics.png)、[独立交付审计](D:/WorkSpace/ZMProject/DataGeneration_work/r35_fusion_tcn_20261028/independent_delivery_audit.json)。

已固定展示前1、3.5、5、10 μm四组，每组有三通道时域/频谱及STFT图。时域仅显示首个1秒窗内400–500 ms；频谱用完整1秒Hann单边幅值；STFT显示0.10–0.90秒、10–900 Hz，输入/重建及各工况使用同一绝对dB参考和色标。九张PNG和SVG的链接均在验收报告。

首次附加兼容检查曾因checkpoint的run_ids不含test而报KeyError，改核对checkpoint配置test_runs后通过；绘图曾因NumPy整数JSON序列化失败，修正后通过；首版曲线遗漏融合RMS失败标记，已补齐并复核。上述修正均未改变物理数据或推理结果。

## 7. 目录、复现与后续边界

- 主线仅放动力学、配置、必要读取/处理/评价、标定和生成。每轮实验、绘图、报告生成及审计脚本只放`tmp/`，输出使用仓库外明确新目录；原始`data/`只读。
- 最新下游脚本位于`tmp/r35_fusion_tcn_20261028/`：prepare_synthesis.py、synthesis.py、prepare_tcn_eval.py、evaluate_tcn.py、plot_results.py、audit_delivery.py。它们依赖明确保存的健康父片段、源码快照和旧权重；复用时先读路径合同，不随意将旧输出重命名为新实验。
- 动力学/β环境：`D:/Softwares/miniconda3/envs/ZMSoftware/python.exe`；TCN环境：`D:/Softwares/miniconda3/envs/spindle-reconstruction/python.exe`。
- 用户最后一次科学任务已完成，当前追加要求是本交接文档。不要自动追加扫描、重新拟合物理参数、改变阈值、重训TCN或因曲线不单调而调合成。
- 尚未建立：最新参数各级h/N和长时认证、可靠的阶次数值不确定度、多真实背景泛化、同步实机回转误差真值。不能把运行内分窗通过、β中位数通过或程序执行完成写成全模型认证。
- 用户最新多代理偏好为GPT-6.1 Terra、Ultra推理强度；遵守不使用sha256、不默认启用Superpowers等规则。本文件是本次授权的交接，不更新AGENTS.md。
