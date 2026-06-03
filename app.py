# -*- coding: utf-8 -*-
"""
YOLOv8 Object Detection App – Notebook‑identical inference
===========================================================
Detects trees, cars, and buildings using two separate YOLOv8 models.
Uses Gradio for the UI.

Models:
  - tree_model.pt        → detects trees (class 0)
  - car_building_model.pt → detects cars (class 0) and buildings (class 1)

Inference matches the notebook: conf=0.30, no extra post‑processing.
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import tempfile
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE MODEL PATHS – replace with your actual file paths
# ─────────────────────────────────────────────────────────────────────────────
TREE_MODEL_PATH        = r"C:\Users\mango\Downloads\best (1).pt"
CAR_BUILDING_MODEL_PATH = r"C:\Users\mango\Downloads\best (3).pt"

# ─────────────────────────────────────────────────────────────────────────────
# Fixed confidence threshold (exactly as used in notebook)
# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.3   # 30%

# ─────────────────────────────────────────────────────────────────────────────
# Color map (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "tree":     (0,   200,  60),   # green
    "car":      (50,  130, 255),   # blue
    "building": (220,  50,  50),   # red
}

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load models (loaded once on first use)
# ─────────────────────────────────────────────────────────────────────────────
_models: dict = {}

def _load_model(path: str, key: str):
    """Load YOLO model and cache it."""
    if key not in _models:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: '{path}'")
        from ultralytics import YOLO
        _models[key] = YOLO(path)
    return _models[key]


# ─────────────────────────────────────────────────────────────────────────────
# Core detection – mimics notebook behaviour exactly
# ─────────────────────────────────────────────────────────────────────────────
def run_detection(pil_image: Image.Image, mode: str):
    """
    Run YOLO inference exactly like notebook:
        model = YOLO('best.pt')
        results = model(image_path)   # default imgsz=640, conf=0.30, iou=0.45
    No extra filters.
    """
    # Select model
    if mode == "Trees only":
        model = _load_model(TREE_MODEL_PATH, "tree")
    else:
        model = _load_model(CAR_BUILDING_MODEL_PATH, "car_building")

    # Save PIL image to a temporary file (allows YOLO to use its native loader)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        pil_image.save(tmp.name)
        img_path = tmp.name

    # Inference – identical to notebook
    results = model(img_path, conf=CONF_THRESHOLD)   # conf only, iou default 0.45

    # Parse detections
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = model.names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class": class_name,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "conf": conf_val,
            })

    # ── Draw bounding boxes (same drawing as before) ────────────────────────
    img_np = np.array(pil_image.convert("RGB"))
    annotated = img_np.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(img_np.shape[1], img_np.shape[0]) / 1200)
    thickness = max(2, int(min(img_np.shape[1], img_np.shape[0]) / 400))

    for det in detections:
        color = COLORS.get(det["class"], (180, 180, 180))
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['class']} {det['conf']*100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            annotated,
            (x1, max(y1 - th - baseline - 6, 0)),
            (x1 + tw + 4, y1),
            color, -1
        )
        cv2.putText(
            annotated, label,
            (x1 + 2, max(y1 - baseline - 4, th)),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    annotated_pil = Image.fromarray(annotated)

    # Table rows
    table_rows = [
        [d["class"], d["x1"], d["y1"], d["x2"], d["y2"], f"{d['conf']:.3f}"]
        for d in detections
    ]

    # Summary counts
    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    if counts:
        summary_lines = [f"**{cls.capitalize()}:** {n}" for cls, n in sorted(counts.items())]
        summary_lines.append(f"\n**Total detections:** {len(detections)}")
        summary_text = "\n".join(summary_lines)
    else:
        summary_text = "⚠️ No objects detected."

    # Save to temp file for download
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(tempfile.gettempdir(), f"detection_{ts}.png")
    annotated_pil.save(save_path)

    return annotated_pil, table_rows, summary_text, save_path


# ─────────────────────────────────────────────────────────────────────────────
# Gradio handler
# ─────────────────────────────────────────────────────────────────────────────
def detect(image, mode):
    if image is None:
        return None, [], "❌ Please upload an image.", None
    try:
        return run_detection(image, mode)
    except Exception as e:
        return None, [], f"❌ Error: {e}", None


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0d1117 !important; font-family: 'Inter', sans-serif; color: #e6edf3; }
#header-box {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 16px; padding: 28px 36px;
    margin-bottom: 8px; text-align: center;
}
#header-box h1 { margin: 0; font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #56d364);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }
#detect-btn { background: linear-gradient(135deg, #238636, #2ea043) !important; border: none !important;
    color: #fff !important; font-weight: 600 !important; border-radius: 8px !important;
    padding: 12px 0 !important; transition: transform 0.15s; }
#detect-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(35,134,54,0.45); }
#save-btn { background: linear-gradient(135deg, #1f6feb, #388bfd) !important; border: none !important;
    color: #fff !important; font-weight: 600 !important; border-radius: 8px !important;
    padding: 10px 0 !important; transition: transform 0.15s; }
#save-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(56,139,253,0.45); }
#legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 4px; }
.chip { display: inline-flex; align-items: center; gap: 6px; background: #21262d !important;
    border: 1px solid #30363d !important; border-radius: 20px !important; padding: 4px 14px !important;
    font-size: 0.82rem !important; font-weight: 600 !important; color: #e6edf3 !important; }
.dot { width: 10px; height: 10px; border-radius: 50%; }
"""

def build_ui():
    with gr.Blocks(title="YOLOv8 Detection App") as demo:
        gr.HTML("""
        <div id="header-box">
            <h1>🔍 YOLOv8 Object Detection</h1>
            <p>Detect <strong>Trees</strong>, <strong>Cars</strong>, and <strong>Buildings</strong>
               from aerial or street-level imagery.</p>
        </div>
        """)
        gr.HTML("""
        <div id="legend">
            <span class="chip"><span class="dot" style="background:#3cc864"></span>Trees</span>
            <span class="chip"><span class="dot" style="background:#3282ff"></span>Cars</span>
            <span class="chip"><span class="dot" style="background:#dc3232"></span>Buildings</span>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙️ Settings")
                image_input = gr.Image(type="pil", label="Upload Image", image_mode="RGB", sources=["upload"], elem_classes=["card"])
                mode_dropdown = gr.Dropdown(choices=["Trees only", "Cars & Buildings"], value="Cars & Buildings", label="Detection Mode")
                detect_btn = gr.Button("🚀 Detect", elem_id="detect-btn", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Annotated Result")
                image_output = gr.Image(type="pil", label="Detection Output", interactive=False, elem_classes=["card"])
                save_btn = gr.DownloadButton("💾 Save Output Image", elem_id="save-btn", visible=False)

        gr.Markdown("### 📋 Detection Details")
        table_output = gr.Dataframe(headers=["Class", "X1", "Y1", "X2", "Y2", "Confidence"],
                                    datatype=["str","number","number","number","number","str"],
                                    interactive=False, wrap=True)
        gr.Markdown("### 📊 Summary")
        summary_output = gr.Markdown(value="Run detection to see results.")
        save_path_state = gr.State(value=None)

        def on_detect(image, mode):
            annotated, table_rows, summary, save_path = detect(image, mode)
            btn_update = gr.update(visible=(save_path is not None), value=save_path)
            return annotated, table_rows, summary, save_path, btn_update

        detect_btn.click(on_detect, [image_input, mode_dropdown],
                         [image_output, table_output, summary_output, save_path_state, save_btn])

        gr.HTML('<p style="text-align:center; color:#484f58;">Powered by Ultralytics YOLOv8 · Gradio</p>')
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  YOLOv8 Detection App (notebook‑identical inference)")
    print("=" * 60)
    print(f"  Tree model        : {TREE_MODEL_PATH}")
    print(f"  Car/Building model: {CAR_BUILDING_MODEL_PATH}")
    print(f"  Fixed confidence  : {CONF_THRESHOLD*100:.0f}%")
    print("=" * 60)
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, css=CSS)