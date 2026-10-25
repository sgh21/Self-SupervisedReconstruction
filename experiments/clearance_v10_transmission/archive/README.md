# V9 归档索引

本目录记录 2026-10-25 的 V9 归档范围和验收边界。归档分支从项目基线单独建立，不移动原工作分支；本次不重跑或修复 V9。

V9 源码、配置和文本证据保留在其原路径：根目录的 `src/spindle/`、`scripts/`、`config/`、`utils/`，以及 `experiments/clearance_v9_tai_bearing/`。本目录另存 `failure_status.json` 和 `archive_manifest_crc32.json`。完整结论见 `docs/reports/clearance_v9_archive_20261025.md`。

已纳入 V9 实验目录中 `.md`、`.json`、`.csv`、`.py`、`.txt`、`.toml`、`.log` 共 927 个文件，合计 37,664,132 字节（35.919 MiB）。最大单文件为收敛对比 JSON，1,264,222 字节；没有超过 50 MiB 的文本文件。归档清单使用 CRC32。原实验 `manifest_crc32.json` 也一并保留。

为避免把大型运行数组上传，实验目录中的 `.npz`（合计 28,487,893,454 字节）、图片、PDF、压缩包及 Numba 缓存不在本分支中。现有 CRC32 清单记录了原实验文件；缺少的原生数组仍需从原工作区取得，不能把本归档当成独立原生数据复核包。

核心结果是部分验收：主工况运行与指定数值加密通过，但通用有限转速失接触连续性失败，20 μm 上端未达到，实机定量验证未完成。详见 `failure_status.json` 和报告中的证据路径。
