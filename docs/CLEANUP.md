# 最小项目整理记录｜2026-10-27

## 范围与实际变化

- 主线Python：**180文件/68910行 → 13文件/约2866行**（格式化可能造成少量行数变化）；删除主线中的旧实验调度、绘图、报告、测试、6/7/8/10自由度入口及V9 Tai支路。
- `src/spindle/`只剩4个动力学文件及`__init__.py`；`src/`3个主程序；根`utils/`4个核心工具及`__init__.py`。不存在`spindle`内标定、数据处理或评价工具。
- `config/`保留完整`model16.json`、`model20.json`和`calibration_targets.json`。两模型所有机械参数、参数来源、适用域、激励、安装实际K/C、求解与采样参数均与整理前一致；删除的是旧任务调度元数据和过期配置，不是当前模型参数。
- `scripts/`从主线移除；原`utils/`全部历史工具移出，根`utils/`只放提取后的主线功能。依赖减少为NumPy、SciPy、Numba；不再导入matplotlib、h5py或旧实验模块。
- `AGENTS.md`置顶主线和目录规则；交接和实验计划分别保存。`.gitignore`允许这几份管理文档，排除`tmp/`、数据、缓存及输出。
- 旧源码/成果移到仓库外`D:/WorkSpace/ZMProject/DataGeneration_archive/20261027_minimal/`，从当前主线移除但留有可回查档案；没有清空硬盘上的历史证据。原始`data/runs`和上游`data/derived_data`保持原位只读。
- 本轮临时迁移、检查代码统一归位`tmp/`，不参与主线导入，不打包提交。未提交/推送Git变更。

## 回归检查与范围

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 两模型物理/求解配置逐项对照 | PASS，相同 | 不修改参数以迁就整理结果 |
| 平衡、质量、实际K/C、测点索引 | PASS | 16：[4,5,15]；20：[4,5,19] |
| 每模型32随机状态，旧/新Python及Numba RHS | 全部逐位相同，最大差0 | 针对当前两种拓扑 |
| Python与Numba交叉 | 最大绝对差16约9.09e−12、20约1.46e−11 | RHS分量含不同单位，逐项容差检查通过，不当统一物理误差条 |
| 两模型0.02 s原生轨迹和接触输出 | 全部逐位相同 | 包括q/v/acc、滚动与配合力、测量位移 |
| 两模型完整480转CLI再生成 | 均完成且运行内稳态通过 | 未追加h/N/时长收敛实验 |
| 完整原生/通道/逐圈数组对上一轮 | 全部逐位相同 | 16共49数组，20共48数组；含NaN不适用标记 |
| 根目录外安装运行 | 已安装editable包；两个最终生成入口及标定入口均实际运行并导出 | 不依赖仓库工作目录或旧实验目录；缩短到32转的检查均未稳态，退出码2 |
| 静态导入/未定义名称检查 | PASS | 不存在绘图、历史脚本及Tai导入依赖 |

完整回归E_dir：16自由度2.402217134571154 μm，20自由度2.176943087089841 μm，与上轮完全一致。

480转回归在核心功能提取后完成；随后按用户要求重排目录，动力学代码未变，重新完成RHS/短轨迹对照及安装后CLI检查，没有为了改目录再重复480转积分。

## 首次问题及未通过项

- 首次目录整理误把主程序/工具放进`src/spindle/`；用户纠正后已按“动力学→spindle，入口→src，工具→utils，临时→tmp”重排并重新检查。
- 缩短到32转、0.6 s热身的RMS标定流程检查只评估1次：fit/val RMS均在[0.5,2]，但稳态未通过；保存`RMS_OR_STEADY_OR_DOMAIN_FAILED`并拒绝输出冻结配置。预算终止，优化器未收敛。此项只验证失败出口，不是生产标定，不覆盖原冻结参数。
- 最终目录的两个生成入口也用32转/0.6 s热身检查，均完整导出但返回NOT_STEADY和退出码2；这是预期的短时稳态未通过，不隐去为PASS。正式480转回归的两组均STEADY，见上表。
- 独立h/2、N1024、960转及局部敏感性未在本轮进行；延续用户之前“不复核”的范围。项目可运行与生产数值认证分别陈述。
- 当前最终25级扫描、β、M1、TCN及新指标趋势仍未完成；见[交接](HANDOVER.md)。

## 证据位置

归档根`D:/WorkSpace/ZMProject/DataGeneration_archive/20261027_minimal/`：

- `cleanup_manifest.json`：原目录→归档目录。
- `verification_mechanics.json`、`verification_final_layout.log`：力学、短轨迹、参数操作及布局复验。
- `verification_delivery.json`：完整数组逐位对照及清理统计。
- `verification/model16/`、`verification/model20/`：完整480转原生状态、通道和逐圈数据。
- `verification/calibration_smoke/`与`verification/final_calibration/`：初次和最终目录下的单次短时标定流程记录。
- `final_cli16.log`、`final_cli20.log`、`final_calibration.log`：最终安装布局、仓库外工作目录的实际入口日志。

上轮三方图文报告现位于`artifacts/results/c1um_rear_fixed_current_comparison_20261027/report/index.html`。由于目录迁移，旧报告中的来源绝对路径仅保留历史含义；本项目README和管理文档均使用新位置。
