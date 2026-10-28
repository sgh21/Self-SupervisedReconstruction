# 20自由度修复与首失败验收报告｜2026-10-27

**执行状态：STOPPED_AT_FIRST_FAILURE。完成计算1级，完整认证通过0级，后24级NOT_RUN。**

第1级 `(F1,F2,R1,R2)=(1,1,2,2) μm` 完成独立静平衡、零初速冷启动及480转原生记录。积分成功、物理域、四个120转子窗的六对物理统计和相干幅值一致性均通过；停止在E层：相干分量可靠性没有独立验证的不确定度模型，仍为INCONCLUSIVE。**这不是已发现动力学失稳，也不是全模型修复成功。**

本轮遵循用户最新更正：20自由度带内RMS分窗门限5%；h/N数值比较仍为2%。其余LIMITS未变。未改质量、实际K/C、激励、预紧或接触本构；未做标定、β、合成或TCN。

## 已修复与仍未解决

| 项目 | 本轮证据与结果 |
|---|---|
| 相干估计误判 | FIXED_CONFIRMED（点估计）：Hann加权联合1–7×/DC，SVD直接求解；保留旧矩形估计。1/3通道、8/120转、幅值及绝对相位恢复通过。 |
| 弱阶次可靠性 | UNRESOLVED：保留全部21个分量、Hann/Hann²差、局部残差及设计矩阵条件数；未知不确定度为null，绝不设为PASS。当前实现尚不能认证任一真实记录的该项。 |
| 首失败停机 | 会话进入时已有局部修复；本轮CLI与专用串行入口回归通过。真实第1级停止，后续目录不存在。 |
| 后间隙默认联动 | FIXED_CONFIRMED：20DOF的with_clearances和CLI均要求显式后间隙；3.25前间隙反例保持后2/2或拒绝省略。 |
| 积分失败落盘 | FIXED_CONFIRMED：partial_native.npz含PARTIAL_NOT_FORMAL；failure.json含求解器字段、源码CRC32、堆栈。原版只有旧文件名和分散诊断，已保留前后证据。 |
| 原生诊断 | 新增接受步绝对时刻；完整轨迹先落盘再做观测。RHS异常只声称保存最后确认checkpoint；拒绝的试算状态不可用。 |

恒幅反例：旧矩形四窗1× RMS=[0.003158465132971401, 0.0026539616927917504, 0.0034500193029726005, 0.00261081545940137] mm/s；新值=[0.0029999220832902657, 0.0030002077927141603, 0.002999731666576458, 0.003000239370562568]。最大真值相对误差0.008944%，预设容差0.01%。同一历史native只重算观测，q/v、E、接触统计和带内RMS逐项不变；历史数组未冒充本轮扫描数据。

## 第1级结果及数值诊断

E_dir/Ex/Ey = **2.175536840 / 2.175536840 / 0.716300171 μm**；Hx/Hy/Py 10–900 Hz RMS = **0.038682739 / 0.035113809 / 0.040719471 mm/s**。最大倾角6.68232e-06 rad，小于0.001 rad。

| 独立诊断 | E_dir差/% | Hx/Hy/Py RMS差/% | 所有冻结特征比较 |
|---|---:|---|---|
| h_half | 4.82392e-05 | 0.000548199 / 0.000386957 / 3.6912e-06 | PASS |
| N1024 | 0.000369258 | 0.0113623 / 0.00310344 / 2.90795e-05 | PASS |
| warmup7 | 1.33306e-05 | 2.57827e-05 / 2.47631e-05 / 2.4533e-07 | PASS |
| long960 | 6.16011e-06 | 1.73277e-06 / 6.82283e-07 / 4.73497e-09 | PASS |

h/2、N1024保持完整初态、绝对时间和相位一致，单独改变各自数值设置。热身7 s保持480转；960转用最后480转做同长度数值比较，另存8个120转窗，避免把双倍时长冲量当误差。所有诊断只针对第1级，置于diagnostics/；没有恢复后续扫描。h/4及Radau未运行：本级未出现需要进一步细化定位的物理非稳态或h/N比较失败时，不作无依据追加。

接触审查：四只配合的几何完全脱开率均为0；配合与滚动接触阻尼残差未见小于−1e−14 W的样本。后滚动阻尼cb_total=0保留。代数能量平衡最大残差约7.42e-16 W；原生网格积分闭合误差占累计耗散0.00122255%。本级没有证据支持负阻尼泵能或接触离散导致非稳态。接触开启律仍非光滑，当前结论仅适用于本级，不外推到旧3.25 μm或未运行等级。

## 逐项验收

| ID | 结果 | 证据/边界 |
|---|---|---|
| V01 来源 | PASS | environment.json、source_before/、source_after/、patch/repair.diff；按仓库规则用CRC32及完整快照，不用SHA256。 |
| V02 人工反例 | PASS点估计 / UNRESOLVED可靠性 | checks/regression.json；不把点估计通过写成真实分量置信区间验证。 |
| V03 物理不变性 | PASS | checks/same_array_invariance.json，含旧/新阶次和六对比较。 |
| V04 首失败mock | PASS | checks/driver_regression.json；首级失败/未知/异常、细化失败均不调用下一级。 |
| V05 积分异常mock | PASS | checks/partial_failure/；checks/before_and_solver_evidence.json覆盖失败返回及RHS异常边界。 |
| V06 后间隙省略 | PASS | CLI启动前报错；with_clearances拒绝隐式联动。 |
| V07 完整第1级 | A/B/C/D PASS，E INCONCLUSIVE | scan/cases/gap_1um/base/；原生、逐圈、每窗和6对比较齐全。 |
| V08 首失败数值诊断 | PASS_FEATURE_COMPARISONS_ONLY | diagnostics/numerics.json、contact/contact_audit.json；不改变首次停机记录。 |
| V09 顺序扫描 | PASS_FAILFAST，非ALL_25_PASS | scan/scan_manifest.csv、first_failure.json；第2–25级NOT_RUN。 |
| V10 交付 | PASS范围内 | 7组PNG/SVG，原生速度、线性单边幅值谱、STFT、回转、接触图及旧/新阶次窗图；全部物理图来自已运行第1级。 |

首次测试失败：checks.py漏建输出目录，未执行动力学；修复后14项回归通过。原失败堆栈保留在checks/regression_failure.json。冒烟32转只验证运行、有限值和数据导出，不当作长时认证。首次正式失败原因完整保留，不因诊断接近而改为PASS。

## 数据与复现

用户收尾确认：本轮结论只写验收报告，不更新AGENTS.md；保留tmp/repair_20261027_r2/下全部本轮脚本供复现。

- `completed_level_features.csv`：本级三通道RMS、峰峰、峰值、峭度和回转指标；`completed_level_orders.csv`：1–7×及旧/新估计。
- `scan/cases/gap_1um/base/native.npz`：原生q/v/acc、接触、位移、状态映射、接受步；signal.npz是保留幅值的处理结果。
- `diagnostics/`：h/2、N1024、热身7 s、960转独立结果；`contact/`保留接触力、耗散和能量数组。
- 源码修改：`src/generation.py`, `src/main.py`, `src/spindle/installation_dynamics.py`, `utils/configuration.py`, `utils/metrics.py`。全部本轮脚本在仓库tmp/repair_20261027_r2/，不成为主线依赖。
- 复现命令及完整证据路径见README.txt。正式扫描必须使用新输出目录和匹配版本的预检；当前观测可靠性未解决，不能从第2级继续。

科学状态：ESTIMATOR_ERROR_CONFIRMED_AND_POINT_ESTIMATE_FIXED；OBSERVATION_RELIABILITY_UNRESOLVED。本轮实际首级未出现原始物理非稳态，故不能据此宣称旧峰附近真实波动根因已定位。

完整结果目录：`D:/WorkSpace/ZMProject/DataGeneration_work/repair_20261027_r2`。
