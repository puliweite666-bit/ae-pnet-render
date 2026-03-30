import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string

from src.ae_pnet_model import AEPNet
from src.metrics import find_peak_pick


HTML_PAGE = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>AE_PNet - P波到时提取（Render版）</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; max-width: 1100px; }
        h1 { margin-bottom: 10px; }
        .box { border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-top: 20px; }
        .result { font-size: 20px; margin-top: 15px; color: #222; font-weight: bold; }
        .sub { margin-top: 8px; color: #444; font-size: 16px; }
        .hint { color: #666; font-size: 14px; margin-top: 6px; }
        .good { color: #0a7f2e; font-weight: bold; }
        .mid { color: #b36b00; font-weight: bold; }
        .bad { color: #c62828; font-weight: bold; }
        input, select, button { margin-top: 10px; padding: 8px; }
        img { margin-top: 20px; max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }
        .row { margin-top: 10px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .mono { font-family: Consolas, monospace; }
    </style>
</head>
<body>
    <h1>AE_PNet - P波到时提取（Render版）</h1>
    <p>上传单条波形文件（支持 .npy / .txt / .csv 单列），自动提取 P 波到时。</p>

    <div class="box">
        <form method="post" enctype="multipart/form-data">
            <div class="grid">
                <div>
                    <label>上传波形文件：</label><br>
                    <input type="file" name="wavefile" required>
                    <div class="hint">公网版不依赖本地 H5，只做波形推理与可选人工对比。</div>
                </div>

                <div>
                    <label>传感器类型：</label><br>
                    <select name="sensor_type">
                        <option value="AUTO" {% if sensor_choice == "AUTO" %}selected{% endif %}>AUTO</option>
                        <option value="P" {% if sensor_choice == "P" %}selected{% endif %}>P</option>
                        <option value="SH" {% if sensor_choice == "SH" %}selected{% endif %}>SH</option>
                        <option value="SV" {% if sensor_choice == "SV" %}selected{% endif %}>SV</option>
                    </select>
                    <div class="hint">AUTO 会根据文件名里的 PU / SL / SR 自动判断。</div>
                </div>

                <div>
                    <label>阈值 threshold：</label><br>
                    <input type="number" name="threshold" min="0" max="1" step="0.05" value="{{ threshold_value }}">
                </div>

                <div>
                    <label>人工 true_pick（可选手动填写）：</label><br>
                    <input type="number" name="true_pick" min="0" step="1" value="{{ true_pick_value }}">
                    <div class="hint">填写后会显示 residual 和拾取质量。</div>
                </div>
            </div>

            <div class="row">
                <button type="submit">开始预测</button>
            </div>
        </form>

        {% if file_name %}
            <div class="sub">文件名：<span class="mono">{{ file_name }}</span></div>
        {% endif %}

        {% if used_sensor_type %}
            <div class="sub">本次使用传感器类型：<strong>{{ used_sensor_type }}</strong></div>
        {% endif %}

        {% if result_text %}
            <div class="result">{{ result_text }}</div>
        {% endif %}

        {% if detail_text %}
            <div class="sub">{{ detail_text }}</div>
        {% endif %}

        {% if quality_text %}
            <div class="{{ quality_class }}" style="margin-top: 10px; font-size: 18px;">{{ quality_text }}</div>
        {% endif %}

        {% if plot_png %}
            <img src="data:image/png;base64,{{ plot_png }}" alt="plot">
        {% endif %}
    </div>
</body>
</html>
"""


def normalize_waveform(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        std = 1.0
    return x / std


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_sensor_type_from_filename(filename: str) -> str:
    name = filename.upper()
    if "PU" in name or "_P" in name:
        return "P"
    if "SL" in name or "SR" in name:
        return "SH"
    return "P"


def load_waveform_from_upload(file_storage, window_size):
    filename = file_storage.filename.lower()

    if filename.endswith(".npy"):
        x = np.load(file_storage)
    else:
        raw = file_storage.read()
        text = raw.decode("utf-8", errors="ignore")
        try:
            x = np.loadtxt(io.StringIO(text), dtype=np.float32, delimiter=",")
        except Exception:
            x = np.loadtxt(io.StringIO(text), dtype=np.float32)

    x = np.asarray(x).reshape(-1).astype(np.float32)
    x = normalize_waveform(x)

    if len(x) > window_size:
        x = x[:window_size]
    elif len(x) < window_size:
        out = np.zeros(window_size, dtype=np.float32)
        out[:len(x)] = x
        x = out

    return x


def make_plot_base64(waveform, p_prob, pred_pick, true_pick=None):
    import base64

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(waveform, label="waveform")
    ax.plot(p_prob, label="p_prob")

    if pred_pick is not None and pred_pick >= 0:
        ax.axvline(pred_pick, linestyle="--", label=f"pred={pred_pick}")

    if true_pick is not None and true_pick >= 0:
        ax.axvline(true_pick, linestyle="-.", label=f"true={true_pick}")

    ax.set_title("P-wave Picking")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def judge_quality(residual_abs: int):
    if residual_abs <= 20:
        return "拾取质量：好", "good"
    elif residual_abs <= 40:
        return "拾取质量：一般", "mid"
    else:
        return "拾取质量：较差", "bad"


def load_model(config_path: str, checkpoint_path: str):
    cfg = load_json(config_path)
    device = "cpu"

    model = AEPNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return cfg, model, device


CONFIG_PATH = os.environ.get("CONFIG_PATH", "config_render.json")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "models/best_model.pth")

cfg, model, device = load_model(CONFIG_PATH, CHECKPOINT_PATH)

window_size = int(cfg["window_size"])
sampling_rate = float(cfg["sampling_rate"])
default_threshold = float(cfg.get("default_threshold", 0.5))

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result_text = None
    detail_text = None
    quality_text = None
    quality_class = ""
    plot_png = None
    file_name = None
    used_sensor_type = None
    sensor_choice = "AUTO"
    threshold_value = default_threshold
    true_pick_value = ""

    if request.method == "POST":
        if "wavefile" not in request.files:
            result_text = "没有检测到上传文件。"
            return render_template_string(
                HTML_PAGE,
                result_text=result_text,
                detail_text=detail_text,
                quality_text=quality_text,
                quality_class=quality_class,
                plot_png=plot_png,
                file_name=file_name,
                used_sensor_type=used_sensor_type,
                sensor_choice=sensor_choice,
                threshold_value=threshold_value,
                true_pick_value=true_pick_value,
            )

        file_obj = request.files["wavefile"]
        file_name = file_obj.filename

        sensor_choice = request.form.get("sensor_type", "AUTO").upper().strip()
        if sensor_choice == "AUTO":
            used_sensor_type = infer_sensor_type_from_filename(file_name)
        else:
            used_sensor_type = sensor_choice

        try:
            threshold_value = float(request.form.get("threshold", default_threshold))
        except Exception:
            threshold_value = default_threshold

        true_pick_raw = request.form.get("true_pick", "").strip()
        true_pick = None
        true_pick_value = true_pick_raw
        if true_pick_raw != "":
            try:
                true_pick = int(true_pick_raw)
            except Exception:
                true_pick = None

        try:
            x = load_waveform_from_upload(file_obj, window_size)

            arr = np.zeros((3, window_size), dtype=np.float32)
            sensor_map = {"P": 0, "SH": 1, "SV": 2}
            arr[sensor_map.get(used_sensor_type, 0)] = x

            with torch.no_grad():
                xt = torch.from_numpy(arr).unsqueeze(0).to(device)
                _, probs = model(xt)
                p_prob = probs[0, 1].detach().cpu().numpy()
                pred_pick = find_peak_pick(p_prob, threshold=threshold_value)
                peak_prob = float(np.max(p_prob))

            if pred_pick >= 0:
                pick_time_us = pred_pick / sampling_rate * 1e6
                result_text = f"检测到 P 波到时：sample={pred_pick}, time={pick_time_us:.3f} 微秒"
            else:
                result_text = "未检测到高于阈值的 P 波到时"

            detail_parts = [f"threshold={threshold_value:.2f}", f"peak_prob={peak_prob:.4f}"]

            if true_pick is not None:
                detail_parts.append(f"true_pick={true_pick}")

            if true_pick is not None and pred_pick >= 0:
                residual = pred_pick - true_pick
                detail_parts.append(f"residual={residual} samples")
                quality_text, quality_class = judge_quality(abs(residual))
            elif true_pick is not None and pred_pick < 0:
                quality_text = "给了 true_pick，但当前阈值下模型没有检出。"
                quality_class = "bad"

            detail_text = "，".join(detail_parts)
            plot_png = make_plot_base64(x, p_prob, pred_pick, true_pick=true_pick)

        except Exception as e:
            result_text = f"处理失败：{e}"

    return render_template_string(
        HTML_PAGE,
        result_text=result_text,
        detail_text=detail_text,
        quality_text=quality_text,
        quality_class=quality_class,
        plot_png=plot_png,
        file_name=file_name,
        used_sensor_type=used_sensor_type,
        sensor_choice=sensor_choice,
        threshold_value=threshold_value,
        true_pick_value=true_pick_value,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
