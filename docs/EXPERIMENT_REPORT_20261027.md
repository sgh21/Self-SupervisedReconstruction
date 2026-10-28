# 20自由度本轮实验结果｜2026-10-27

[打开完整图文报告（12张对比/趋势图）](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/report/index.html) · [完整Markdown报告](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/report/report.md)

| 方面 | 结果 |
|---|---|
| 模型与参数 | 健康三通道RMS的fit/val六个比值均达[0.5,2]；1 μm独立数值复核通过。3.25、10 μm细化及960转仍未稳态。 |
| 25级数据 | 四间隙同步1–10 μm，步长0.375 μm；25/25完整波形、逐圈标签和实际配置。14级稳态，11级未稳态。 |
| β与单背景合成 | 本轮1 μm重新标定β，15/21项启用；一个留出背景生成25条条件M1，1 μm严格零注入。严格分支仅1 μm可用。 |
| TCN | 真实健康按run划分25/3/14，从头训练100轮；25条全部推理。有效14级的E_dir与relative_mse相关约0.284，按E_dir排序有7次分数下降，不能支持单调精度预测。 |

合成幅值外推较大：有效5.875 μm样本ACC1平均窗口RMS约为训练最大窗口的62.8倍；异常分数尚无真实故障强度或实机回转误差校准。数值未通过点、未做逐点预算以及历史数据非全新盲测均在完整报告单列。

完整产物在`D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/`：

- [机械25级汇总](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/scan/scan_summary.csv)：各目录含`native.npz`、`signal.npz`、逐圈标签、实际配置和稳态证据。
- [β和合成证据](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/beta_m1/delivery_summary.json) · [25条条件M1清单](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/beta_m1/conditional_final/manifest.csv)。
- [TCN训练/评分证据](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/tcn/delivery_summary.json) · [最终25级评分CSV](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/tcn/m1_strict_ieee/levels.csv)。
- [交付完整性检查](D:/WorkSpace/ZMProject/DataGeneration_work/pipeline_20261027/report/delivery_integrity.json)：25级输入、数组、标签和run隔离核验通过；该项不覆盖稳态/数值失败。

本轮临时脚本统一在`tmp/pipeline_20261027/`，未进入主线导入依赖。当前物理参数保持冻结；没有根据TCN趋势调整β或机械参数。
