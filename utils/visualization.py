"""Fixed validation cases: physical waveforms, shared FFT spectra, and HTML."""

import json
from pathlib import Path

import h5py
from matplotlib.figure import Figure
import numpy as np
import torch

from utils.reconstruction import CHANNELS, amplitude_spectrum, reconstruction_metrics
from utils.spindle_dataset import REPO_ROOT


def select_cases(dataset, sample_count=3):
    """Choose evenly spaced runs and their middle available window, without RNG."""
    if sample_count < 1:
        raise ValueError("visualization.sample_count must be positive")
    by_run = {}
    for index, (file_index, _) in enumerate(dataset.samples):
        by_run.setdefault(file_index, []).append(index)
    run_indices = list(by_run)
    positions = np.linspace(0, len(run_indices) - 1, min(sample_count, len(run_indices)), dtype=int)
    cases = []
    for position in positions:
        indices = by_run[run_indices[position]]
        dataset_index = indices[len(indices) // 2]
        file_index, window_index = dataset.samples[dataset_index]
        path = dataset.files[file_index]
        with h5py.File(path, "r") as h5:
            cases.append({
                "run_id": dataset.run_ids[file_index],
                "dataset_index": dataset_index,
                "window_index": window_index,
                "source_file": path.relative_to(REPO_ROOT).as_posix(),
                "start_time_s": float(h5["windows_1s/start_time_s"][window_index]),
                "end_time_s": float(h5["windows_1s/end_time_s"][window_index]),
            })
    return cases


def _comparison_figure(original, reconstructed, frequency, original_spectrum,
                       reconstructed_spectrum, metrics, sample_rate_hz, title):
    figure = Figure(figsize=(13, 8), layout="constrained")
    axes = figure.subplots(3, 2)
    time_s = np.arange(original.shape[-1]) / sample_rate_hz
    for channel, name in enumerate(CHANNELS):
        waveform, spectrum = axes[channel]
        waveform.plot(time_s, original[channel], color="#2864a0", lw=1.0, label="Input")
        waveform.plot(time_s, reconstructed[channel], color="#d85d32", lw=0.9,
                      ls="--", label="Reconstruction")
        waveform.set(xlabel="Time within window (s)", ylabel="Velocity (mm/s)",
                     xlim=(0, original.shape[-1] / sample_rate_hz))
        waveform.set_title(
            f"{name} | RMSE {metrics['time_rmse_mm_s'][channel]:.4g} mm/s"
            f" | NRMSE {metrics['time_nrmse'][channel]:.2%}", fontsize=10,
        )
        spectrum.plot(frequency, original_spectrum[channel], color="#2864a0", lw=1.0, label="Input")
        spectrum.plot(frequency, reconstructed_spectrum[channel], color="#d85d32", lw=0.9,
                      ls="--", label="Reconstruction")
        spectrum.set(xlabel="Frequency (Hz)", ylabel="Amplitude (mm/s)", xlim=(10, 900))
        spectrum.set_ylim(bottom=0)
        spectrum.set_title(f"{name} | Hann amplitude spectrum, 10-900 Hz", fontsize=10)
        for axis in (waveform, spectrum):
            axis.grid(alpha=0.18)
            axis.tick_params(labelsize=9)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(title, fontsize=13)
    return figure, axes


def _write_html(directory, manifest):
    # Embedded metadata lets the report open directly from disk without a server.
    data = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>验证集重建案例</title><link rel="icon" href="data:,">
<style>
body{font-family:system-ui,sans-serif;color:#172c42;background:#f3f6fa;margin:0}
main{max-width:1400px;margin:28px auto;padding:0 24px}h1{margin-bottom:8px}
p{line-height:1.7}label{display:inline-block;margin:8px 18px 8px 0}
select{padding:8px;border:1px solid #c2ceda;border-radius:6px;background:white}
img{width:100%;height:auto;background:white;border-radius:8px}
table{border-collapse:collapse;background:white;margin:16px 0;width:100%}
th,td{padding:9px 14px;border-bottom:1px solid #dde4eb;text-align:left}
.muted{color:#526678}
</style><main><h1>验证集重建案例</h1>
<p id="model"></p><p class="muted">蓝色：输入；橙色虚线：重构。波形已还原为 mm/s。
频谱沿用评价指标的 Hann 窗与 rFFT 幅值定义，频带为 10–900 Hz。
案例按 run 固定选取，不代表整个验证集的平均误差。
所有误差指标按完整 1 秒窗口计算，局部放大只改变波形显示范围。</p>
<label>权重阶段 <select id="stage"></select></label>
<label>采集案例 <select id="sample"></select></label>
<label>时域范围 <select id="view"><option value="image">完整窗口</option>
<option value="zoom_image">局部放大</option></select></label>
<p id="caption"></p><img id="figure" alt="三通道输入与重构的波形及频谱对比">
<table><thead><tr><th>通道</th><th>时域 RMSE (mm/s)</th><th>时域 NRMSE</th>
<th>频谱 NRMSE</th></tr></thead><tbody id="metrics"></tbody></table></main>
<script>
const report=__DATA__;
const stage=document.querySelector('#stage'),sample=document.querySelector('#sample'),view=document.querySelector('#view');
document.querySelector('#model').textContent=`模型：${report.model} ｜ 数据：验证集`;
report.stages.forEach((s,i)=>stage.add(new Option(s.name==='best'?`最佳权重 · 第 ${s.epoch} 轮`:`第 ${s.epoch} 轮`,i)));
stage.value=String(report.stages.length-1);
function render(){
 const s=report.stages[Number(stage.value)],c=s.cases[Number(sample.value)];
 document.querySelector('#figure').src=c[view.value];
 document.querySelector('#caption').textContent=`${c.run_id} ｜ run 内时间 ${c.start_time_s.toFixed(3)}–${c.end_time_s.toFixed(3)} s ｜ 第 ${s.epoch} 轮`;
 const table=document.querySelector('#metrics');table.replaceChildren();
 ['ACC1','ACC2','ACC3'].forEach((name,i)=>{
  const row=table.insertRow();
  [name,c.metrics.time_rmse_mm_s[i].toPrecision(5),(100*c.metrics.time_nrmse[i]).toFixed(3)+'%',(100*c.metrics.spectrum_nrmse[i]).toFixed(3)+'%'].forEach(v=>row.insertCell().textContent=v);
 });
}
function changeStage(){
 const previous=sample.value;sample.replaceChildren();
 report.stages[Number(stage.value)].cases.forEach((c,i)=>sample.add(new Option(c.run_id,i)));
 if(previous && Number(previous)<sample.options.length)sample.value=previous;
 render();
}
stage.onchange=changeStage;sample.onchange=render;view.onchange=render;changeStage();
</script></html>""".replace("__DATA__", data)
    (directory / "index.html").write_text(html, encoding="utf-8")


@torch.no_grad()
def save_reconstruction_cases(model, dataset, cases, normalizer, device, use_amp,
                              sample_rate_hz, output_dir, epoch, model_name,
                              writer=None, best=False, zoom_time_s=(0.2, 0.3)):
    """Render a few fixed validation windows; no training or full-set evaluation."""
    directory = Path(output_dir) / "visualizations"
    stage = "best" if best else f"epoch_{epoch:03d}"
    stage_dir = directory / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    original = torch.stack([dataset[case["dataset_index"]] for case in cases]).to(device)
    center, scale = normalizer["center"].to(device), normalizer["scale"].to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        prediction = model((original - center) / scale)
    reconstructed = (prediction.float() * scale + center).cpu()
    original = original.cpu()
    model.train(was_training)
    metrics = reconstruction_metrics(original, reconstructed, sample_rate_hz)
    frequency, original_spectrum = amplitude_spectrum(original, sample_rate_hz)
    _, reconstructed_spectrum = amplitude_spectrum(reconstructed, sample_rate_hz)
    rendered = []
    for index, case in enumerate(cases):
        case_id = f"{case['run_id']}_window_{case['window_index']:04d}"
        case_metrics = {key: value[index].tolist() for key, value in metrics.items()}
        title = (f"{model_name} | Validation | {'Best checkpoint, ' if best else ''}epoch {epoch}\n"
                 f"{case['run_id']} | run time {case['start_time_s']:.3f}-{case['end_time_s']:.3f} s")
        figure, axes = _comparison_figure(
            original[index].numpy(), reconstructed[index].numpy(), frequency.numpy(),
            original_spectrum[index].numpy(), reconstructed_spectrum[index].numpy(),
            case_metrics, sample_rate_hz, title,
        )
        image = f"{stage}/{case_id}.png"
        zoom_image = f"{stage}/{case_id}_zoom.png"
        figure.savefig(directory / image, dpi=140)
        tag = f"{'validation_best' if best else 'validation'}/{case_id}"
        if writer is not None:
            writer.add_figure(f"{tag}/full", figure, global_step=epoch, close=False)
        for axis in axes[:, 0]:
            axis.set_xlim(*zoom_time_s)
        figure.suptitle(title + f" | Time zoom {zoom_time_s[0]:g}-{zoom_time_s[1]:g} s", fontsize=13)
        figure.savefig(directory / zoom_image, dpi=140)
        if writer is not None:
            writer.add_figure(f"{tag}/zoom", figure, global_step=epoch, close=False)
        figure.clear()
        rendered.append({**case, "image": image, "zoom_image": zoom_image, "metrics": case_metrics})
    manifest_path = directory / "index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "model": model_name, "split": "validation", "sample_rate_hz": sample_rate_hz, "stages": [],
    }
    manifest["stages"] = [entry for entry in manifest["stages"] if entry["name"] != stage]
    manifest["stages"].append({"name": stage, "epoch": epoch, "cases": rendered})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(directory, manifest)
    return directory / "index.html"
