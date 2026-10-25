# V9 复现说明

本文记录当前代码可用的复现入口。运行结果必须写入新的、唯一的输出目录；不要覆盖 `data_A`、`data_A_long`、`data_B` 或既有验证证据。本文只列命令，不表示本轮重新运行。

## 环境与当前旧激励

在仓库根目录使用项目环境：

```powershell
$py = 'D:\Softwares\miniconda3\envs\ZMSoftware\python.exe'
$env:PYTHONPATH = '.\src'
```

当前可复现旧激励的配置是 `config/clearance_v9_config.json`。其中 `U_kg_m=2.689135290615865e-5`、相位 `−0.3661687497 rad`、转速 `−7000 rpm`；命令中不传 `--U`，保留配置值。该配置用于 old-U 机制对照，不是新模型健康再标定配置。V9 runner 未显式传 `--turns` 时会把保留窗设为 480 转；下面显式写出 480 转和 A 组使用的 3.5 s 初始热身，便于复现条件。

例如重跑 N3 五个间隙点：

```powershell
$run = 'experiments/clearance_v9_tai_bearing/repro_runs/oldU_N3_YYYYMMDD_HHMMSS'
& $py -m scripts.run_clearance_v9 `
  --config config/clearance_v9_config.json `
  --output $run --group data_A --variant N3 `
  --gaps 2 3 5 7.5 10 --turns 480 --warmup 3.5 --workers 2
```

每个机制变体需单独调用 runner，`--variant` 可设 `B0`、`B0d`、`N1`、`N2` 或 `N3`。N1/N2/N3 的差异见[模型范围](reports/model_scope.md)。`--output` 是运行根目录，例中案例写入 `$run/data_A/N3/gap_*um/`，并保留实际配置、输入、积分检查点、接受步长、原生状态和诊断。把 `YYYYMMDD_HHMMSS` 换成尚未使用的唯一标记。需要续算时 runner 提供 `--resume`，一次只续一个间隙点；先核对对应检查点和原 `actual_config.json`。

V9健康确认已完成：原U的健康指标1.99003647 μm，通过独立h/根/N/采样检查；U±10%的实际试算均通过稳态，响应分别1.77813660/2.20170170 μm。因此U_new保持2.689135290615865e−5 kg·m，冻结于`config/parameters_v9_calibrated.json`；源码内同一内容位于`config/clearance_v9_calibrated.json`，标定历史见`calibration/history.csv`。2026-10-25 19:15:59（UTC+8）完成冻结。B组c2/c5/c10复用同输入A轨迹，3/7.5 μm在冻结后新增运行，具体定位看`case_selection.json`和`data_B/reuse_manifest.json`。

从仓库根重跑全部B五点时，把上述命令的配置改为`config/clearance_v9_calibrated.json`、`--group`改为`data_B`，并使用新的输出目录。复现是新的独立积分，不覆盖复用的原A证据。

## 离线分析、验收和交付

各工具的真实 CLI 已通过 `--help` 核对；以下操作都应指向刚生成的目录或新的输出目录。

```powershell
# 从原生 case 重放信号/套圈/逐球分析，输出 case_summary.csv 与 offline_analysis.json
& $py -m utils.clearance_v9_analysis --root $run --group data_A `
  --replay --output "$run/recomputed"

# 按 selection 读取已完成的 A/B 案例生成图
& $py -m utils.clearance_v9_figures --root $run --output "$run/figures"

# 离线汇总验收；selection 必须列出实际案例目录和细化对照
& $py -m utils.clearance_v9_acceptance `
  --selection '<已冻结的case_selection.json>' --root . `
  --output "$run/acceptance"

# 可选的独立轴承/短整机重验；这会执行数值验证，不是离线整理
& $py -m utils.clearance_v9_validation `
  --contract GPTOrder/spindle_v9_tai_bearing_plan/templates/v9_contract.json `
  --system-config config/clearance_v9_config.json `
  --output-dir "$run/validation"

# 汇总现有证据并记录环境；--review-only 仅生成轻量 review 包
& $py -m utils.clearance_v9_delivery --root $run `
  --package --review-only --destination "deliverables/spindle_v9_YYYYMMDD_HHMMSS"
```

验收工具需要明确的 `case_selection.json`；不要把失败、未通过的加密项或轨迹来源从 selection 中悄悄移除。Delivery 只盘点和打包实际存在的证据，完成积分不自动等于数值或物理验收通过。`clearance_v9_validation` 会跑单轴承和短时整机检查；它不是长时矩阵 runner。Acceptance、分析、绘图和打包命令不会启动长时积分。

## 资源与复现边界

runner 的 `--workers` 是并行案例数，默认 2；当前工作机为 32 GB 物理内存，可用约 31.77 GiB，另有 20 个逻辑处理器。V9 每个 worker 会保留原生 `q/v/a`、接受步长及求根/套圈诊断，内存与磁盘写入会随并发增加。建议从 2 workers 起步，依据实际峰值内存和磁盘余量再调整，不按逻辑 CPU 数直接拉满。当前没有覆盖全部变体的峰值内存测量，因此不声称更高并发已安全或更快。

复现结果以各案例的 `actual_config.json`、`actual_solver_settings.json`、`actual_input.npz`、`environment/actual_environment.json` 和原生输出为准。配置、依赖、求解器版本、热身续接或接触分支不同，都应作为不同运行记录；不要把重新运行覆盖到原证据目录。

## 轻量审阅包与复算边界

本次交付使用 `--package --review-only`：review ZIP 收录源码、配置、报告、图表及现有 CSV/JSON 诊断；排除 NPZ、NPY、XZ、PDF、ZIP 文件，不生成 native 或 supplemental 数据包。包内不含原始波形、原生轨迹数组、独立加密数组或失败状态数组，也不在包内运行分析重放、验收重算、接触边界复算或失败状态回放。此包面向代码与结果审阅；包内源码可供后续独立运行，但原生轨迹复核须另行取得完整证据与相应配置。

从仓库根生成轻量包时使用新的目的目录；本次最终清单与压缩包大小以 [spindle_v9_packages.json](../../deliverables/spindle_v9/spindle_v9_packages.json) 为准。此处不预填体积，也不表示本说明执行过打包命令：

```powershell
$run = 'experiments/clearance_v9_tai_bearing'
& $py -m utils.clearance_v9_delivery --root $run `
  --package --review-only --destination 'deliverables/spindle_v9_review_YYYYMMDD_HHMMSS'
```

最终选择包含17个A/B案例角色，对应14条不同主轨迹；B组c2/c5/c10复用A组N3轨迹，c3/c7.5为冻结后新增积分。最终盘点为48个已保存案例目录（含4条继承轨迹来源和9条明确记录续接来源），另保留1个失败长算目录。压缩包旁的 `spindle_v9_packages.json` 和包内 `manifest_crc32.json` 记录交付文件清单、字节数及 CRC32；清单中的排除说明不代表本地文件已删除。完整磁盘证据仍保留在实验根目录，Delivery 盘点会更新 `summary.csv`、`run_inventory.json`、`environment/actual_environment.json` 和 `manifest_crc32.json`。

如需后续复算，先从本地备份或另行交付取得完整原始证据根目录（包括对应案例 `native.npz`；失败回放还需失败状态、初值及所需旧 solver 快照）。以下仅示例如何从包内代码读取外部证据；不会读取包内数据，也不属于本次交付步骤：

```powershell
$pkg = '<解压目录>/spindle_v9_review'
$dataRoot = '<另行取得的完整 V9 实验数据根目录>'
Set-Location $pkg
$env:PYTHONPATH = './source;./source/src'
& $py -B -m utils.clearance_v9_analysis --root $dataRoot --group data_A `
  --replay --output './recomputed_external/data_A'
```

要重放其他组或独立加密项，需另行提供相应完整证据并分别指定组、selection 和新的输出目录。此包本身不能完成原生轨迹复核；本文没有运行包内或外部数据复算。
