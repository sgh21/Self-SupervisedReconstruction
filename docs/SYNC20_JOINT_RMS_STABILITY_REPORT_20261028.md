> **2026-10-28最新留存整理：** 用户要求仅留存最后一轮及其复现依赖；删除旧实验与临时脚本的操作被自动审批阻止，当前尚未清理，清理清单已保存。请先查看 [最终留存与复现说明](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/README_留存与复现.md) 和 [历史记录](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/历史记录.md)。

# 同步20级联合实验验收索引｜2026-10-28

状态：**已完成；按最新规则，来源分窗RMS为19/20通过。** 25套候选、125个搜索工况、20级正式仿真、20条融合波形及旧TCN的80窗推理全部完成。除来源分窗RMS外不设任何验收门槛。

[当前口径图文报告与图片](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/rms_only/report.md)｜[原任务历史记录](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/report.md)｜[任务书](TASKBOOK_SYNC20_JOINT_RMS_STABILITY_20261028.md)

## 联合选参

历史搜索从原20自由度实际K＋16自由度实际C出发，八项实际K/C各相对原始起点限制在±20%。起点＋12套覆盖＋12套局部候选，每套同时尝试精确同步1、3.5、4、4.5、5 μm五工况，按当时的预登记规则固定c07。该选择未按最新规则重新执行或重新选参。

- 起点H/S/J：0.055066931 / 2.745787820 / 2.800854752。
- 最终H/S/J：0.052572409 / 0 / 0.052572409；只称预算内选定结果，不宣称收敛或全局最优。
- 健康fit比值：[1.2443, 1.0292, 0.7187]；冻结后val：[1.5424, 1.0703, 1.3326]，仅为数值报告，不设比值合格线。
- 四个指定点整体RMS全部通过，最坏相对差3.0841%。

生效输入为独立保存的[final_config.json](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/final_config.json)。根`config/model20.json`的动力学参数仍是此前c24，已移除旧筛查阈值；后续复现不能与c07混用。

## 正式20级及下游验收

正式网格为`g_i=5i/19，i=0…19`，四间隙显式同步。精确1 μm及四个搜索点单独登记；正式5 μm经配置、源码、冷启动和原生数组一致性校验后复用。固定参数后未根据扫描或TCN结果重选。

| 层次 | 实际结果 |
|---|---|
| 完整仿真导出 | 20/20；独立冷启动，3.5 s热身＋480转 |
| 来源整体RMS分窗 | 19/20；4.210526 μm处Hx最大差6.1147%未通过 |
| 模型E_dir范围 | 0.381757–12.227362 μm，原生M−P、保留1× |
| 新β | 训练10 run/30父片段拟合；结果保留，不作为额外验收门槛 |
| 固定健康背景融合 | 20/20完成；不检查或标记融合RMS |
| 精确1 μm健康对照 | 零注入，与父数组逐位恒等；额外4窗单独保存 |
| 旧TCN | 20级80窗完成；沿用旧权重和训练Z-score，无重训；MSE/不去均值余弦独立复算通过 |

20级E_dir与MSE的描述性Spearman=0.94436，与余弦=-0.91429。MSE最大0.503776 (mm/s)²、余弦最低0.505642都在3.421053 μm，曲线仍非单调。这是单背景条件下的结果。

## 图、数据与执行记录

- [20级RMS与回转误差](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/rms_only/figures/generation_summary.png)、[TCN指标](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/rms_only/figures/tcn_metrics_relative_health.png)，PNG均有同名SVG。
- 代表等级在查看TCN前固定为网格索引[4,13,17,19]；各有健康/融合短时域与线性幅值谱、输入/重建叠加、共同绝对色标STFT，见完整报告。
- [搜索证据](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/search/search_summary.json)、[正式仿真清单](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/generation/levels.csv)、[融合清单](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/synthesis/manifest.csv)、[逐级指标](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/tcn/levels.csv)、[输入/重建数组](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/tcn/reconstructions.npz)。
- 原运行的首次失败和修复记录保留在历史报告；历史融合RMS计算不再作为数据验收或绘图标记。
- 调度从4到8进程在完整工况边界交接，全部既有结果复用，无重复积分；未改物理参数边界、数值设置或验收门限。

结果根目录：`D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/`。用户已确认：结论只写验收报告，不更新AGENTS.md；保留`tmp/sync20_joint_rms_stability_20261028/`全部本轮临时脚本供复现，不自动清理，不移入主线，不追加实验。

追加重画：[相对健康MSE（对数轴）与1−余弦](D:/WorkSpace/ZMProject/DataGeneration_work/sync20_joint_rms_stability_20261028/figures/tcn_metrics_relative_health.png)。健康分母为本轮精确四隙1 μm零注入对照的四窗平均MSE `9.324875804850308e−7 (mm/s)²`，保留全部20级，仅标记来源分窗RMS超差。
