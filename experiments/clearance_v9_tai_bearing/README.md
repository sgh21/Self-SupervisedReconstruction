# V9独立复核入口

先读[实际报告](report.md)和`acceptance.json`，再看[来源核对](source_audit.md)、[参数变化](parameter_delta.csv)与[复现说明](reproducibility.md)。模型保持12个结构自由度、前后背靠背安装及原壳体模型，分别计算内外滚道接触与套圈力/矩。

本次对外交付通过 `utils.clearance_v9_delivery --package --review-only` 生成轻量复核包，包含代码、配置、报告、图表及现有 CSV/JSON 诊断和验收摘要；不包含原始波形、原生轨迹数组、独立加密或失败状态 NPZ，也排除 NPY、XZ、PDF 和 ZIP 来源文件。包内不执行原生重放或复算。完整磁盘证据仍保留在本实验目录；打包过滤不删除或迁移这些文件。实际包名及字节数以最终清单[spindle_v9_packages.json](../../deliverables/spindle_v9/spindle_v9_packages.json)为准。

**通用有限速失接触切换连续性未通过。** 持续双接触主工况的数值验收与该失败项分别列出；不能把主工况运行成功当作通用接触切换或实机验证通过。原失败状态和修复后复算都保留。

## 文件定位

| 内容 | 位置 |
|---|---|
| 正式A/B点与独立加密的确切路径 | `case_selection.json` |
| 健康确认及U邻域试算 | `calibration/history.csv`、`calibration/health_confirmation.json` |
| 冻结输入 | `config/parameters_v9_calibrated.json` |
| 同旧U机制分解（原生数据仅保留在实验目录） | `data_A/`、`data_A_long/` |
| 冻结U扫描及A数据复用声明 | `data_B/reuse_manifest.json` |
| 独立h/根/N加密证据（NPZ仅保留在实验目录） | `convergence/` |
| 源公式、原文与借用参数 | `references/`、`task_contract/`、`source_audit.md` |
| 原V8源码与A起始源码 | `source_before/`、`source_at_A/` |
| 原生记录目录盘点 | `summary.csv`、`run_inventory.json` |
| 数值、初值、边界及失败回放 | `validation/` |
| 可重绘图、完整工况表 | `figures/` |
| 文件完整性与环境 | `manifest_crc32.json`、`environment/` |

健康确认保留原U，故B的2、5、10 μm直接引用A的同输入480转轨迹；3、7.5 μm为冻结后新增积分。它们共同组成B五点曲线，复用不是新增独立运行。120转先导与480转续算都保留，正式点以选择文件为准。

## 轻量包与复算边界

解压清单中的 review 包即可阅读代码、配置、报告、图表、验收摘要和来源记录。该轻量包不带原始波形、原生轨迹、独立加密NPZ或失败状态NPZ，因此不能单独执行`clearance_v9_analysis --replay`或失败状态重放；本次交付也不执行包内复算。不要把本机完整实验目录误认为已打入压缩包。

打包仅排除文件，不删除实验目录中已有的`data_A/`、`data_A_long/`、`data_B/`、`convergence/`、`sensitivity/`等原生或加密证据。若后续另行取得完整磁盘证据，才可从解压包中的代码指向外部数据根目录做可选复算；例如：

```powershell
$pkg = '<解压目录>/spindle_v9_review'
$dataRoot = '<另行取得的完整V9实验数据根目录>'
Set-Location $pkg
$env:PYTHONPATH = './source;./source/src'
python -B -m utils.clearance_v9_analysis --root $dataRoot --group data_A `
  --replay --output './recomputed_external/data_A'
```

按`data_A_long`、`data_B`或`convergence`复算时，分别改`--group`并使用不同输出目录。完整复算会读取外部`native.npz`并重算逐球诊断；失败回放还需另行提供对应失败状态和初值NPZ。以上均是可选的外部数据复算示例，不属于本次轻量交付步骤，也没有在交付包上执行。

主回转指标是固定探头参考下，逐转X/Y峰峰值各取均值后取较大者，保留1×。H主比较频带为10–900 Hz，使用原生状态速度。静态中心偏移、单转最大峰峰值和相对健康增量分别报告，不互相替代。
