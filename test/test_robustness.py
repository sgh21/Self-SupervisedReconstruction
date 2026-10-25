"""Evaluate saved TCN baselines and denoising models on paired external noise tests."""

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from model.tcn import build_autoencoder
from utils.augmentation import add_noise
from utils.reconstruction import CHANNELS, amplitude_spectrum
from utils.robustness import evaluate_robustness
from utils.spindle_dataset import REPO_ROOT, SpindleDataset
from utils.visualization import select_cases


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return "Infinity" if isinstance(value, float) and value == math.inf else value


def summarize(rows):
    """Pool squared errors over noise seeds before calculating RMSE and SNR."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"])].append(row)
    summary = []
    for (model, condition), values in groups.items():
        for channel in CHANNELS:
            reference_rms = values[0]["reference_rms_mm_s"][channel]
            mse = sum(row["rmse_mm_s"][channel] ** 2 for row in values) / len(values)
            noise_mse = sum(reference_rms ** 2 * 10 ** (-row["input_snr_db"][channel] / 10)
                            for row in values) / len(values)
            summary.append({
                "model": model, "condition": condition, "noise_type": values[0]["noise_type"],
                "snr_db": values[0]["snr_db"], "channel": channel, "noise_seeds": len(values),
                "sample_count": values[0]["sample_count"], "rmse_mm_s": math.sqrt(mse),
                "reference_rms_mm_s": reference_rms,
                "input_snr_db": 10 * math.log10(reference_rms ** 2 / noise_mse) if noise_mse else math.inf,
                "output_snr_db": 10 * math.log10(reference_rms ** 2 / mse) if mse else math.inf,
            })
    return summary


def _run_rows(rows):
    flattened = []
    for row in rows:
        for run_id, metrics in row["run_metrics"].items():
            for channel in CHANNELS:
                flattened.append({
                    "model": row["model"], "condition": row["condition"], "seed": row["seed"],
                    "noise_seed": row["noise_seed"],
                    "run_id": run_id, "channel": channel, "sample_count": metrics["sample_count"],
                    "mse_normalized": metrics["mse_normalized"],
                    **{key: metrics[key][channel] for key in (
                        "rmse_mm_s", "reference_rms_mm_s", "input_snr_db", "output_snr_db",
                    )},
                })
    return flattened


@torch.no_grad()
def save_examples(models, dataset, normalizers, device, use_amp, rate, seeds, output):
    """Examples share noise across models and have separately recorded example seeds."""
    directory = output / "examples"
    directory.mkdir(exist_ok=True)
    examples = [("clean", None), ("white", 10), ("pink", 10), ("narrowband", 10),
                ("impulse", 10), ("pink+impulse", 0)]
    manifest = []
    for case_index, case in enumerate(select_cases(dataset, 2)):
        reference = dataset[case["dataset_index"]].unsqueeze(0).to(device)
        for condition_index, (kind, snr) in enumerate(examples):
            seed = seeds[0] + 1000000 + case_index * 1000 + condition_index
            corrupted = reference if kind == "clean" else add_noise(
                reference, kind, snr, rate, torch.Generator(device=device).manual_seed(seed),
            )
            signals = {"reference": reference.cpu()[0], "noisy": corrupted.cpu()[0]}
            for label, model in models.items():
                center = normalizers[label]["center"].to(device)
                scale = normalizers[label]["scale"].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    prediction = model((corrupted - center) / scale)
                signals[label] = (prediction.float() * scale + center).cpu()[0]
            frequency, _ = amplitude_spectrum(signals["reference"], rate)
            spectra = {label: amplitude_spectrum(signal, rate)[1] for label, signal in signals.items()}
            time = np.arange(reference.shape[-1]) / rate
            condition = "clean" if snr is None else f"{kind}_{snr}dB"
            for architecture in ("custom", "standard"):
                baseline, augmented = f"{architecture}_baseline", f"{architecture}_augmented"
                legend_labels = {"reference": "Reference", "noisy": "Corrupted",
                                 baseline: "Baseline", augmented: "Augmented"}
                figure = Figure(figsize=(13, 8), layout="constrained")
                axes = figure.subplots(3, 2)
                for channel, name in enumerate(CHANNELS):
                    for label, color, style in (
                        ("reference", "#2864a0", "-"), ("noisy", "#999999", "-"),
                        (baseline, "#d85d32", "--"), (augmented, "#23935b", "--"),
                    ):
                        axes[channel, 0].plot(time, signals[label][channel], color=color, ls=style,
                                              lw=0.9, alpha=0.5 if label == "noisy" else 1, label=legend_labels[label])
                        axes[channel, 1].plot(frequency, spectra[label][channel], color=color,
                                              ls=style, lw=0.9, alpha=0.5 if label == "noisy" else 1, label=legend_labels[label])
                    axes[channel, 0].set(xlim=(0, reference.shape[-1] / rate), xlabel="Time within window (s)",
                                         ylabel="Velocity (mm/s)", title=name)
                    axes[channel, 1].set(xlim=(10, 900), ylim=(0, None), xlabel="Frequency (Hz)",
                                         ylabel="Hann amplitude (mm/s)", title=name)
                    for axis in axes[channel]:
                        axis.grid(alpha=0.2)
                        axis.legend(fontsize=7)
                figure.suptitle(f"{architecture} | {condition} | {case['run_id']}\n"
                                f"Independent example seed {seed}; run time {case['start_time_s']:.3f} s")
                filename = f"{architecture}_{case['run_id']}_{condition.replace('+', '_')}.png"
                figure.savefig(directory / filename, dpi=140)
                for axis in axes[:, 0]:
                    axis.set_xlim(0.2, 0.3)
                zoom_filename = filename.removesuffix(".png") + "_zoom.png"
                figure.savefig(directory / zoom_filename, dpi=140)
                figure.clear()
                manifest.append({**case, "architecture": architecture, "noise_type": kind,
                                 "snr_db": snr, "example_seed": seed, "image": f"examples/{filename}",
                                 "zoom_image": f"examples/{zoom_filename}",
                                 "realization": "representative independently seeded realization"})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _report(summary, rows, manifest, output, metadata):
    lookup = {(row["model"], row["condition"], row["channel"]): row for row in summary}
    conditions = list(dict.fromkeys(row["condition"] for row in summary))
    report_config = metadata["config"]
    training_description = report_config.get(
        "training_description", "增强模型沿用既有 TCN 权重做去噪微调；方案选择只使用 7 月验证数据。",
    )
    test_history_description = report_config.get(
        "test_history_description",
        "该外部数据未参与这些 TCN 的训练/验证，但曾用于其他特征分析，不能称为全新盲测。",
    )
    lines = ["# TCN 人工干扰测试", "", "指标以未叠加人工干扰的物理信号为参考，所有模型使用相同测试窗口及相同噪声。",
             "在原始 mm/s 量纲按每个窗口、每个通道 RMS 定标，再使用各自 checkpoint 的固定训练归一化。",
             training_description, "",
             f"测试 run：{', '.join(metadata['test_run_ids'])}。{test_history_description}"
             f"本次包含 {len(metadata['test_run_ids'])} 个完整 run，不把窗口或人工噪声重复当独立机械试验。",
             "多噪声种子先汇总平方误差，再计算 RMSE 和输出 SNR。完整通道指标见 summary.csv，逐 run/种子见 per_run.csv。", ""]
    for architecture in ("custom", "standard"):
        baseline, augmented = f"{architecture}_baseline", f"{architecture}_augmented"
        lines += [f"## {architecture}", "", "| 条件 | 基线 RMSE ACC1/2/3 (mm/s) | 增强 RMSE ACC1/2/3 | 基线输出 SNR ACC1/2/3 (dB) | 增强输出 SNR ACC1/2/3 (dB) | SNR 增益 ACC1/2/3 (dB) |", "|---|---|---|---|---|---|"]
        gains = []
        no_net_recovery = []
        for condition in conditions:
            b = [lookup[(baseline, condition, channel)] for channel in CHANNELS]
            a = [lookup[(augmented, condition, channel)] for channel in CHANNELS]
            gain = [new["output_snr_db"] - old["output_snr_db"] for old, new in zip(b, a)]
            baseline_errors = " / ".join(f"{value['rmse_mm_s']:.4g}" for value in b)
            augmented_errors = " / ".join(f"{value['rmse_mm_s']:.4g}" for value in a)
            baseline_snr = " / ".join(f"{value['output_snr_db']:.2f}" for value in b)
            augmented_snr = " / ".join(f"{value['output_snr_db']:.2f}" for value in a)
            lines.append(f"| {condition} | {baseline_errors} | {augmented_errors} | "
                         f"{baseline_snr} | {augmented_snr} | "
                         f"{' / '.join(f'{value:+.2f}' for value in gain)} |")
            if condition != "clean":
                gains.append((condition, sum(gain) / len(gain)))
                no_net_recovery.extend(
                    f"{condition}/{channel}" for channel, value in zip(CHANNELS, a)
                    if value["output_snr_db"] <= value["input_snr_db"]
                )
        clean_ratios = [lookup[(augmented, "clean", c)]["rmse_mm_s"] / lookup[(baseline, "clean", c)]["rmse_mm_s"] for c in CHANNELS]
        lines += ["", "干净信号 RMSE 比值（增强/基线）：" + " / ".join(f"{value:.3f}" for value in clean_ratios) + "。大于 1 表示重建退化。",
                  f"受扰条件中，三通道平均输出 SNR 增益为正的条件：{sum(g > 0 for _, g in gains)}/{len(gains)}。",
                  "仍未改善的条件：" + (", ".join(name for name, gain in gains if gain <= 0) or "无；各通道差异仍需查看表格") + "。", ""]
        lines += ["增强输出 SNR 未超过受扰输入 SNR（没有净恢复收益）的条件/通道："
                  + (", ".join(no_net_recovery) or "无") + "。高输入 SNR 下，固有重建误差也可能限制净收益。", ""]
        per_run = defaultdict(list)
        for row in rows:
            if row["model"] == augmented:
                for run_id, metrics in row["run_metrics"].items():
                    per_run[run_id].extend(metrics["output_snr_db"].values())
        lines.append("各 run 的输出 SNR 范围（涵盖全部条件/种子/通道）：" + "; ".join(f"{run}: {min(v):.2f}–{max(v):.2f} dB" for run, v in per_run.items()) + "。")
    lines += ["", "## 代表性案例", "", "以下图片为独立记录种子的代表性实现；同图的基线与增强模型收到完全相同受扰输入。",
              "FFT 复用已有 Hann/rFFT 幅值函数。图片展示完整 1 s 波形，并另存 _zoom.png 局部放大版本；指标按整个窗口计算。", ""]
    lines += [f"- [{entry['architecture']} / {entry['run_id']} / {entry['noise_type']} / {entry['snr_db']} dB]({entry['image']})" for entry in manifest]
    lines += ["", "人工白噪声、有色噪声、窄带或脉冲叠加仅用于仿真压力测试；真实冲击、谐波或边带可能是机械特征，不能因此当噪声删除。",
              "结果只说明这些条件下的重建抗干扰能力，不代表真实现场故障检测有效性，也不提供机械故障真值。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["test_data"]
    root = REPO_ROOT / data["root"]
    files = sorted({path for pattern in data["run_patterns"] for path in root.glob(pattern)})
    dataset = SpindleDataset(files, data["sample_rate_hz"], data["input_length"])
    device = torch.device(config["device"])
    use_amp = device.type == "cuda" and config["amp"]
    models, normalizers, checkpoint_metadata = {}, {}, {}
    for label, path in config["models"].items():
        checkpoint = torch.load(REPO_ROOT / path, map_location="cpu", weights_only=True)
        used = set(checkpoint["run_ids"]["train"] + checkpoint["run_ids"]["validation"])
        if used.intersection(dataset.run_ids):
            raise ValueError(f"Test runs overlap training/validation in {label}")
        for key in ("sample_rate_hz", "input_length"):
            if checkpoint["config"]["data"][key] != data[key]:
                raise ValueError(f"Test {key} differs from {label}")
        models[label] = build_autoencoder(**checkpoint["config"]["model"]).to(device)
        models[label].load_state_dict(checkpoint["model_state"])
        normalizers[label] = checkpoint["normalizer"]
        checkpoint_metadata[label] = {"checkpoint": str((REPO_ROOT / path).resolve()),
                                      "epoch": checkpoint["epoch"], "run_ids": checkpoint["run_ids"],
                                      "normalization": {
                                          "mode": checkpoint["normalizer"]["mode"],
                                          "mean": checkpoint["normalizer"]["center"].reshape(-1).tolist(),
                                          "std": checkpoint["normalizer"]["scale"].reshape(-1).tolist(),
                                      },
                                      "config": checkpoint["config"]}
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"],
                        pin_memory=device.type == "cuda", persistent_workers=config["num_workers"] > 0)
    scenarios = [{"name": "clean", "noise_type": "clean", "snr_db": None}] + [
        {"name": f"{kind}_{snr}dB", "noise_type": kind, "snr_db": snr}
        for kind in config["noise_types"] for snr in config["snr_dbs"]
    ]
    output = REPO_ROOT / config["output_dir"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Independent test: {len(dataset)} windows; runs {dataset.run_ids}; output {output}", flush=True)
    rows = evaluate_robustness(models, loader, normalizers, device, use_amp, data["sample_rate_hz"],
                               scenarios, config["seeds"], progress_desc="Paired interference test")
    scenario_indices = {scenario["name"]: index for index, scenario in enumerate(scenarios)}
    for row in rows:
        row["noise_seed"] = (None if row["seed"] is None else
                             row["seed"] + scenario_indices[row["condition"]] * 100003)
    metadata = {"test_run_ids": dataset.run_ids, "source_files": [str(path.resolve()) for path in files],
                "checkpoints": checkpoint_metadata, "config": config,
                "noise_generator_rule": "base seed + scenario index * 100003; clean input SNR is Infinity"}
    (output / "metrics.json").write_text(
        json.dumps(_json_value({**metadata, "rows": rows}), indent=2, allow_nan=False), encoding="utf-8",
    )
    _csv(output / "per_run.csv", _run_rows(rows))
    summary = summarize(rows)
    _csv(output / "summary.csv", summary)
    manifest = save_examples(models, dataset, normalizers, device, use_amp,
                             data["sample_rate_hz"], config["seeds"], output)
    _report(summary, rows, manifest, output, metadata)
    print(f"Saved report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
