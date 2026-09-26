#!/usr/bin/env python3
"""Manual OCR annotation tool for the 160 YOLO training cases.

The annotation JSON uses the contest coordinate convention: origin at the
lower-left, x increases rightward, and y increases upward.  A second set of
PaddleOCR labels is rebuilt automatically; PaddleOCR uses the usual image
coordinate convention with origin at the upper-left.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tkinter as tk
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "赛题六公开数据集" / "200_train_cases"
DEFAULT_OUTPUT = ROOT / "datasets" / "ocr_manual_160"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="默认排除当前 40 张验证图；指定此项后加载全部 200 张",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查并打印待标注图片，不打开图形界面",
    )
    parser.add_argument("--web", action="store_true", help="使用浏览器标注模式")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ocr-device", default="cpu", help="框选后的 OCR 设备：cpu 或 gpu:0")
    parser.add_argument(
        "--ocr-model",
        default="PP-OCRv6_small_rec",
        choices=("PP-OCRv6_tiny_rec", "PP-OCRv6_small_rec", "PP-OCRv6_medium_rec"),
        help="框选后的文字识别模型；手动框选时不需要文本检测模型",
    )
    parser.add_argument(
        "--page-ocr-score",
        type=float,
        default=0.20,
        help="整页 medium OCR 预标注的最低识别分数；最终仍需人工核对",
    )
    parser.add_argument(
        "--ocr-upscale",
        type=float,
        default=1.5,
        help="识别前放大裁剪文字，适合很小的字；设为 1 禁用放大",
    )
    parser.add_argument("--no-ocr-assist", action="store_true", help="关闭框选后的自动 OCR")
    return parser.parse_args()


def current_validation_ids() -> set[str]:
    validation_dir = ROOT / "outputs" / "yolov13_components" / "single_validation"
    return {
        path.stem.removesuffix("_original")
        for path in validation_dir.glob("*_original.png")
    }


def collect_samples(source: Path, include_validation: bool) -> list[dict[str, Any]]:
    samples = []
    validation_ids = current_validation_ids()
    for case_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        case_id = case_dir.name
        if not include_validation and case_id in validation_ids:
            continue
        images = sorted(
            path
            for path in case_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if len(images) != 1:
            raise SystemExit(f"{case_dir} 中应有且仅有一张图片，实际为 {len(images)} 张")
        samples.append({"case_id": case_id, "source": images[0]})
    if not samples:
        raise SystemExit(f"没有找到可标注图片：{source}")
    return samples


def lower_left_point(point: tuple[float, float], image_height: int) -> list[float]:
    return [round(float(point[0]), 2), round(float(image_height - point[1]), 2)]


def top_left_point(point: list[float], image_height: int) -> list[float]:
    return [float(point[0]), float(image_height - point[1])]


def lower_left_bbox(points: list[list[float]], image_height: int) -> dict[str, float]:
    top_points = [top_left_point(point, image_height) for point in points]
    xs = [point[0] for point in top_points]
    ys = [point[1] for point in top_points]
    return {
        "x1": round(min(xs), 2),
        "y1": round(float(image_height - max(ys)), 2),
        "x2": round(max(xs), 2),
        "y2": round(float(image_height - min(ys)), 2),
    }


def annotation_path(output: Path, case_id: str) -> Path:
    return output / "annotations" / f"{case_id}.json"


def load_annotation(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ocr_result_field(result: Any, field: str, default: Any) -> Any:
    try:
        return result[field]
    except (KeyError, TypeError, IndexError):
        pass
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        # PaddleX 的结果通常把字段放在 res 中；兼容不同版本的返回结构。
        if isinstance(payload.get("res"), dict):
            payload = payload["res"]
        return payload.get(field, default)
    return default


def quad_order(points: list[list[float]]) -> list[list[float]]:
    """Return TL, TR, BR, BL order for a four-point polygon."""
    if len(points) != 4:
        raise ValueError("透视裁剪需要四个点")
    # Standard sum/difference ordering is robust for rotated text boxes.
    top_left = min(points, key=lambda point: point[0] + point[1])
    bottom_right = max(points, key=lambda point: point[0] + point[1])
    top_right = min(points, key=lambda point: point[1] - point[0])
    bottom_left = max(points, key=lambda point: point[1] - point[0])
    return [top_left, top_right, bottom_right, bottom_left]


def perspective_crop(image_path: Path, points_top_left: list[list[float]], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    if len(points_top_left) == 4:
        points = quad_order(points_top_left)
        src = np.float32(points)
        width = max(
            int(round(((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2) ** 0.5)),
            int(round(((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2) ** 0.5)),
            1,
        )
        height = max(
            int(round(((points[3][0] - points[0][0]) ** 2 + (points[3][1] - points[0][1]) ** 2) ** 0.5)),
            int(round(((points[2][0] - points[1][0]) ** 2 + (points[2][1] - points[1][1]) ** 2) ** 0.5)),
            1,
        )
        dst = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(image, matrix, (width, height))
    else:
        xs = [point[0] for point in points_top_left]
        ys = [point[1] for point in points_top_left]
        crop = image[int(min(ys)) : int(max(ys)) + 1, int(min(xs)) : int(max(xs)) + 1]
    if crop.size == 0 or not cv2.imwrite(str(output_path), crop):
        raise ValueError(f"无法保存 OCR 裁剪图：{output_path}")


class OCRAnnotator:
    def __init__(self, args: argparse.Namespace, samples: list[dict[str, Any]]):
        self.args = args
        self.samples = samples
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "annotations").mkdir(exist_ok=True)
        (self.output / "images").mkdir(exist_ok=True)
        (self.output / "paddleocr").mkdir(exist_ok=True)
        (self.output / "rec_crops").mkdir(exist_ok=True)

        self.index = 0
        self.image: Image.Image | None = None
        self.image_path: Path | None = None
        self.image_width = 0
        self.image_height = 0
        self.annotations: list[dict[str, Any]] = []
        self.scale = 1.0
        self.fit_scale = 1.0
        self.mode = "rect"
        self.drag_start: tuple[float, float] | None = None
        self.polygon_points: list[tuple[float, float]] = []
        self.selected_tid: str | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("赛题六 OCR 手工标注工具")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<KeyPress-r>", lambda _event: self.set_mode("rect"))
        self.root.bind("<KeyPress-p>", lambda _event: self.set_mode("polygon"))
        self.root.bind("<KeyPress-n>", lambda _event: self.next_sample())
        self.root.bind("<KeyPress-b>", lambda _event: self.previous_sample())
        self.root.after(100, self.load_sample)

    def build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")
        self.status = ttk.Label(toolbar, text="准备加载")
        self.status.pack(side="left", padx=8)
        ttk.Button(toolbar, text="上一张", command=self.previous_sample).pack(side="right")
        ttk.Button(toolbar, text="下一张", command=self.next_sample).pack(side="right", padx=4)
        ttk.Button(toolbar, text="保存", command=self.save_current).pack(side="right", padx=4)
        ttk.Button(toolbar, text="删除选中", command=self.delete_selected).pack(side="right", padx=4)
        ttk.Button(toolbar, text="四点框", command=lambda: self.set_mode("polygon")).pack(side="right", padx=4)
        ttk.Button(toolbar, text="矩形框", command=lambda: self.set_mode("rect")).pack(side="right", padx=4)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#262626", cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        ybar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        xbar = ttk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        ybar.pack(side="right", fill="y")
        xbar.pack(side="bottom", fill="x")
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Button-3>", self.cancel_shape)
        self.canvas.bind("<MouseWheel>", self.zoom_event)
        self.canvas.bind("<Button-4>", self.zoom_event)
        self.canvas.bind("<Button-5>", self.zoom_event)

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.drag_start = None
        self.polygon_points = []
        self.redraw()
        self.set_status(f"模式：{'四点框' if mode == 'polygon' else '矩形框'}")

    def source_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * self.scale, point[1] * self.scale

    def event_to_source(self, event: tk.Event) -> tuple[float, float]:
        return self.canvas.canvasx(event.x) / self.scale, self.canvas.canvasy(event.y) / self.scale

    def clamp_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            min(max(0.0, point[0]), float(self.image_width)),
            min(max(0.0, point[1]), float(self.image_height)),
        )

    def load_sample(self) -> None:
        sample = self.samples[self.index]
        source = Path(sample["source"])
        target = self.output / "images" / f"{sample['case_id']}{source.suffix.lower()}"
        if not target.exists():
            shutil.copy2(source, target)
        self.image_path = target
        self.image = Image.open(target).convert("RGB")
        self.image_width, self.image_height = self.image.size
        data = load_annotation(annotation_path(self.output, sample["case_id"]))
        self.annotations = list(data.get("texts", {}).values()) if data else []
        self.selected_tid = None
        self.scale = 1.0
        self.root.update_idletasks()
        canvas_w = max(400, self.canvas.winfo_width() - 20)
        canvas_h = max(300, self.canvas.winfo_height() - 20)
        self.fit_scale = min(canvas_w / self.image_width, canvas_h / self.image_height, 1.0)
        self.scale = self.fit_scale
        self.redraw()
        self.set_status(self.status_text())

    def status_text(self) -> str:
        case_id = self.samples[self.index]["case_id"]
        return f"{self.index + 1}/{len(self.samples)}  case={case_id}  已标注={len(self.annotations)}  模式={'四点' if self.mode == 'polygon' else '矩形'}"

    def redraw(self) -> None:
        if self.image is None:
            return
        width = max(1, int(round(self.image_width * self.scale)))
        height = max(1, int(round(self.image_height * self.scale)))
        display = self.image.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        for annotation in self.annotations:
            points = annotation.get("points_top_left", [])
            if not points:
                continue
            canvas_points = [self.source_to_canvas((p[0], p[1])) for p in points]
            flat = [coord for point in canvas_points for coord in point]
            color = "#00e676" if annotation.get("tid") == self.selected_tid else "#00b0ff"
            self.canvas.create_polygon(*flat, outline=color, width=2, fill="", tags=(annotation.get("tid", ""),))
            x, y = canvas_points[0]
            self.canvas.create_text(x + 4, y - 4, anchor="sw", text=f"{annotation.get('tid')} {annotation.get('content')}", fill=color, font=("Arial", 12, "bold"))
        if self.polygon_points:
            points = [self.source_to_canvas(point) for point in self.polygon_points]
            flat = [coord for point in points for coord in point]
            self.canvas.create_line(*flat, fill="#ffcc00", width=2)
            for point in points:
                self.canvas.create_oval(point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4, fill="#ffcc00")

    def annotation_at(self, point: tuple[float, float]) -> dict[str, Any] | None:
        x, y = point
        for annotation in reversed(self.annotations):
            points = annotation.get("points_top_left", [])
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                return annotation
        return None

    def mouse_down(self, event: tk.Event) -> None:
        point = self.clamp_point(self.event_to_source(event))
        selected = self.annotation_at(point)
        if selected:
            self.selected_tid = selected["tid"]
            self.redraw()
            self.set_status(f"已选中 {selected['tid']}：{selected['content']}，按 Delete 删除")
            return
        if self.mode == "polygon":
            self.polygon_points.append(point)
            self.redraw()
            if len(self.polygon_points) == 4:
                self.finish_shape(self.polygon_points)
        else:
            self.drag_start = point

    def mouse_move(self, event: tk.Event) -> None:
        if self.mode != "rect" or self.drag_start is None:
            return
        self.redraw()
        start = self.source_to_canvas(self.drag_start)
        end = self.source_to_canvas(self.event_to_source(event))
        self.canvas.create_rectangle(*start, *end, outline="#ffcc00", width=2)

    def mouse_up(self, event: tk.Event) -> None:
        if self.mode != "rect" or self.drag_start is None:
            return
        start = self.drag_start
        end = self.clamp_point(self.event_to_source(event))
        self.drag_start = None
        x1, x2 = sorted((max(0.0, start[0]), min(float(self.image_width), end[0])))
        y1, y2 = sorted((max(0.0, start[1]), min(float(self.image_height), end[1])))
        if x2 - x1 >= 3 and y2 - y1 >= 3:
            self.finish_shape([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        else:
            self.redraw()

    def cancel_shape(self, _event: tk.Event) -> None:
        self.drag_start = None
        self.polygon_points = []
        self.redraw()

    def finish_shape(self, points: list[tuple[float, float]]) -> None:
        content = simpledialog.askstring(
            "输入文字",
            "请输入框内完整文字（严格区分大小写，不要输入 pin_ 前缀）：",
            parent=self.root,
        )
        self.polygon_points = []
        if content is None or content == "":
            self.redraw()
            return
        tid = f"T_{len(self.annotations) + 1}"
        top_points = [[round(float(x), 2), round(float(y), 2)] for x, y in points]
        lower_points = [lower_left_point(point, self.image_height) for point in points]
        self.annotations.append(
            {
                "tid": tid,
                "content": content,
                "points": lower_points,
                "points_top_left": top_points,
                "location": lower_left_bbox(lower_points, self.image_height),
            }
        )
        self.selected_tid = tid
        self.save_current()

    def delete_selected(self) -> None:
        if not self.selected_tid:
            return
        self.annotations = [item for item in self.annotations if item.get("tid") != self.selected_tid]
        self.selected_tid = None
        self.save_current()

    def zoom_event(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if getattr(event, "num", None) == 4 or delta > 0:
            self.scale = min(self.scale * 1.2, 8.0)
        elif getattr(event, "num", None) == 5 or delta < 0:
            self.scale = max(self.scale / 1.2, self.fit_scale)
        self.redraw()

    def save_current(self) -> None:
        if self.image is None or self.image_path is None:
            return
        case_id = self.samples[self.index]["case_id"]
        payload = {
            "case_id": case_id,
            "image": {
                "file_name": f"images/{self.image_path.name}",
                "width": self.image_width,
                "height": self.image_height,
            },
            "coordinate_system": {
                "origin": "lower_left",
                "x_direction": "right",
                "y_direction": "up",
                "units": "pixel",
            },
            "texts": {item["tid"]: dict(item) for item in self.annotations},
        }
        path = annotation_path(self.output, case_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.rebuild_training_files()
        self.set_status(self.status_text() + "  已自动保存")

    def rebuild_training_files(self) -> None:
        det_rows = []
        rec_rows = []
        crop_dir = self.output / "rec_crops"
        expected_crops: set[str] = set()
        for path in sorted((self.output / "annotations").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            image_info = data["image"]
            image_rel = Path(image_info["file_name"])
            items = list(data.get("texts", {}).values())
            if not items:
                continue
            det_items = []
            image_path = self.output / image_rel
            for item in items:
                top_points = item.get("points_top_left")
                if top_points is None:
                    top_points = [top_left_point(point, image_info["height"]) for point in item["points"]]
                det_items.append({"transcription": item["content"], "points": top_points})
                crop_name = f"{data['case_id']}__{item['tid']}.png"
                crop_path = crop_dir / crop_name
                perspective_crop(image_path, top_points, crop_path)
                expected_crops.add(crop_name)
                rec_rows.append(f"../rec_crops/{crop_name}\t{item['content']}")
            det_rows.append(f"../{image_rel.as_posix()}\t{json.dumps(det_items, ensure_ascii=False)}")
        for stale in crop_dir.glob("*.png"):
            if stale.name not in expected_crops:
                stale.unlink()
        paddle_dir = self.output / "paddleocr"
        (paddle_dir / "det_label.txt").write_text("\n".join(det_rows) + ("\n" if det_rows else ""), encoding="utf-8")
        (paddle_dir / "rec_gt.txt").write_text("\n".join(rec_rows) + ("\n" if rec_rows else ""), encoding="utf-8")
        manifest = {
            "coordinate_system": "contest_lower_left_in_annotations; paddleocr_top_left_in_labels",
            "annotated_images": len(list((self.output / "annotations").glob("*.json"))),
            "text_instances": len(rec_rows),
            "det_label": "paddleocr/det_label.txt",
            "rec_label": "paddleocr/rec_gt.txt",
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def next_sample(self) -> None:
        self.save_current()
        if self.index < len(self.samples) - 1:
            self.index += 1
            self.load_sample()
        else:
            messagebox.showinfo("完成", "已经到达最后一张图片。")

    def previous_sample(self) -> None:
        self.save_current()
        if self.index > 0:
            self.index -= 1
            self.load_sample()

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.save_current()
        self.root.destroy()


def run_web_server(args: argparse.Namespace, samples: list[dict[str, Any]]) -> None:
    """Run the same annotator through a browser, suitable for headless servers."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "paddleocr").mkdir(exist_ok=True)
    (output / "rec_crops").mkdir(exist_ok=True)
    sample_map: dict[str, dict[str, Any]] = {}
    for sample in samples:
        source = Path(sample["source"])
        target = output / "images" / f"{sample['case_id']}{source.suffix.lower()}"
        if not target.exists():
            shutil.copy2(source, target)
        image = Image.open(target)
        sample_map[sample["case_id"]] = {
            **sample,
            "target": target,
            "width": image.width,
            "height": image.height,
        }

    ocr_model: Any = None
    page_ocr_model: Any = None

    def get_ocr_model() -> Any:
        nonlocal ocr_model
        if args.no_ocr_assist:
            return None
        if ocr_model is None:
            import os as _os

            _os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".paddlex"))
            from paddleocr import TextRecognition

            print(
                f"正在加载 OCR 识别模型，model={args.ocr_model}, "
                f"device={args.ocr_device}"
            )
            ocr_model = TextRecognition(
                model_name=args.ocr_model,
                device=args.ocr_device,
                enable_mkldnn=False,
                engine="paddle",
            )
        return ocr_model

    def get_page_ocr_model() -> Any:
        """Load the full-page detector/recognizer only when preannotation is requested."""
        nonlocal page_ocr_model
        if args.no_ocr_assist:
            return None
        if page_ocr_model is None:
            import os as _os

            _os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".paddlex"))
            from paddleocr import PaddleOCR

            print(
                "正在加载整页 OCR 预标注模型，model=PP-OCRv6_medium, "
                f"device={args.ocr_device}"
            )
            page_ocr_model = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_det_limit_side_len=2000,
                text_det_limit_type="max",
                text_det_box_thresh=0.20,
                text_rec_score_thresh=0.0,
                device=args.ocr_device,
                enable_mkldnn=False,
                engine="paddle",
            )
        return page_ocr_model

    def top_left_rect_from_polygon(polygon: Any) -> list[list[float]] | None:
        if hasattr(polygon, "tolist"):
            polygon = polygon.tolist()
        if not isinstance(polygon, list) or len(polygon) < 4:
            return None
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return None
            points.append((float(point[0]), float(point[1])))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return [
            [round(x1, 2), round(y1, 2)],
            [round(x2, 2), round(y1, 2)],
            [round(x2, 2), round(y2, 2)],
            [round(x1, 2), round(y2, 2)],
        ]

    def rect_iou(first: list[list[float]], second: list[list[float]]) -> float:
        first_x = [point[0] for point in first]
        first_y = [point[1] for point in first]
        second_x = [point[0] for point in second]
        second_y = [point[1] for point in second]
        ix1, iy1 = max(min(first_x), min(second_x)), max(min(first_y), min(second_y))
        ix2, iy2 = min(max(first_x), max(second_x)), min(max(first_y), max(second_y))
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        first_area = max(0.0, max(first_x) - min(first_x)) * max(0.0, max(first_y) - min(first_y))
        second_area = max(0.0, max(second_x) - min(second_x)) * max(0.0, max(second_y) - min(second_y))
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def full_page_preannotate(sample: dict[str, Any]) -> dict[str, Any]:
        path = annotation_path(output, sample["case_id"])
        existing = load_annotation(path) or empty_payload(sample)
        preannotation = existing.get("preannotation") or {}
        if preannotation.get("status") == "completed":
            return existing
        ocr = get_page_ocr_model()
        if ocr is None:
            raise RuntimeError("整页 OCR 已被 --no-ocr-assist 关闭")

        results = list(ocr.predict(input=str(sample["target"])))
        result = results[0] if results else None
        texts = dict(existing.get("texts", {}))
        used_tids = set(texts)
        next_number = 1
        while f"T_{next_number}" in used_tids:
            next_number += 1
        existing_rects = [item.get("points_top_left", []) for item in texts.values()]
        rec_texts = ocr_result_field(result, "rec_texts", []) if result is not None else []
        rec_scores = ocr_result_field(result, "rec_scores", []) if result is not None else []
        rec_polys = ocr_result_field(result, "rec_polys", None) if result is not None else None
        if rec_polys is None and result is not None:
            rec_polys = ocr_result_field(result, "dt_polys", [])
        rec_texts = rec_texts.tolist() if hasattr(rec_texts, "tolist") else rec_texts
        rec_scores = rec_scores.tolist() if hasattr(rec_scores, "tolist") else rec_scores
        rec_polys = rec_polys.tolist() if hasattr(rec_polys, "tolist") else rec_polys
        added = 0
        for number, raw_text in enumerate(rec_texts or []):
            content = str(raw_text or "").strip()
            if not content:
                continue
            score = None
            if isinstance(rec_scores, list) and number < len(rec_scores):
                try:
                    score = float(rec_scores[number])
                except (TypeError, ValueError):
                    score = None
            if score is not None and score < args.page_ocr_score:
                continue
            if not isinstance(rec_polys, list) or number >= len(rec_polys):
                continue
            top_points = top_left_rect_from_polygon(rec_polys[number])
            if top_points is None:
                continue
            if any(rect_iou(top_points, old) >= 0.5 for old in existing_rects if len(old) == 4):
                continue
            tid = f"T_{next_number}"
            next_number += 1
            used_tids.add(tid)
            existing_rects.append(top_points)
            lower = [lower_left_point((point[0], point[1]), sample["height"]) for point in top_points]
            texts[tid] = {
                "tid": tid,
                "content": content,
                "points": lower,
                "points_top_left": top_points,
                "location": lower_left_bbox(lower, sample["height"]),
                "source": "auto",
                "score": round(score, 4) if score is not None else None,
            }
            added += 1
        payload = empty_payload(sample)
        payload["texts"] = texts
        payload["preannotation"] = {
            "status": "completed",
            "model": "PP-OCRv6_medium",
            "score_threshold": args.page_ocr_score,
            "count": added,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    def recognize_box(sample: dict[str, Any], points_top_left: list[list[float]]) -> dict[str, Any]:
        ocr = get_ocr_model()
        if ocr is None:
            return {"text": "", "score": None}
        assist_dir = output / ".ocr_assist"
        assist_dir.mkdir(exist_ok=True)
        crop_path = assist_dir / f"{sample['case_id']}.png"
        perspective_crop(sample["target"], points_top_left, crop_path)
        try:
            if args.ocr_upscale > 1.0:
                crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                if crop is not None:
                    crop = cv2.resize(
                        crop,
                        None,
                        fx=args.ocr_upscale,
                        fy=args.ocr_upscale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    cv2.imwrite(str(crop_path), crop)
            results = list(ocr.predict(input=str(crop_path), batch_size=1))
            if not results:
                return {"text": "", "score": None}
            result = results[0]
            text = str(ocr_result_field(result, "rec_text", "") or "").strip()
            score_value = ocr_result_field(result, "rec_score", None)
            score = float(score_value) if score_value is not None else None
            return {
                "text": text,
                "score": round(score, 4) if score is not None else None,
            }
        finally:
            crop_path.unlink(missing_ok=True)

    def empty_payload(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": sample["case_id"],
            "image": {
                "file_name": f"images/{sample['target'].name}",
                "width": sample["width"],
                "height": sample["height"],
            },
            "coordinate_system": {
                "origin": "lower_left",
                "x_direction": "right",
                "y_direction": "up",
                "units": "pixel",
            },
            "texts": {},
            "preannotation": None,
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: Any, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = (ROOT / "ocr_annotator_web.html").read_bytes()
                self.send_bytes(html, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/samples":
                result = []
                for sample in samples:
                    path = annotation_path(output, sample["case_id"])
                    count = 0
                    if path.is_file():
                        count = len(json.loads(path.read_text(encoding="utf-8")).get("texts", {}))
                    result.append({"case_id": sample["case_id"], "image": f"/images/{sample['case_id']}{Path(sample['source']).suffix.lower()}", "count": count})
                self.send_json(result)
                return
            if parsed.path == "/api/annotation":
                case_id = parse_qs(parsed.query).get("case", [""])[0]
                sample = sample_map.get(case_id)
                if sample is None:
                    self.send_json({"error": "unknown case"}, 404)
                    return
                path = annotation_path(output, case_id)
                self.send_json(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else empty_payload(sample))
                return
            if parsed.path.startswith("/images/"):
                name = Path(parsed.path.removeprefix("/images/")).name
                sample = next((item for item in sample_map.values() if item["target"].name == name), None)
                if sample is None or not sample["target"].is_file():
                    self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)
                    return
                content_type = "image/png" if sample["target"].suffix.lower() == ".png" else "image/jpeg"
                self.send_bytes(sample["target"].read_bytes(), content_type)
                return
            self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/recognize":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    sample = sample_map[str(body["case_id"])]
                    points = body["points_top_left"]
                    if not isinstance(points, list) or len(points) != 4:
                        raise ValueError("需要四个坐标点")
                    self.send_json(recognize_box(sample, points))
                except Exception as exc:
                    self.send_json({"error": str(exc), "text": "", "score": None}, 500)
                return
            if parsed.path == "/api/rebuild":
                try:
                    writer = OCRAnnotator.__new__(OCRAnnotator)
                    writer.output = output
                    writer.rebuild_training_files()
                    self.send_json({"ok": True, "message": "OCR 训练标签已重建"})
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                return
            if parsed.path == "/api/auto_annotate":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    sample = sample_map[str(body["case_id"])]
                    payload = full_page_preannotate(sample)
                    self.send_json(payload)
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                return
            if parsed.path != "/api/save":
                self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                case_id = str(body["case_id"])
                sample = sample_map[case_id]
                raw_texts = body.get("texts", {})
                existing = load_annotation(annotation_path(output, case_id)) or {}
                texts: dict[str, Any] = {}
                for tid, item in raw_texts.items():
                    content = item.get("content")
                    top_points = item.get("points_top_left")
                    if not isinstance(content, str) or content == "":
                        raise ValueError(f"{tid} 的文字为空")
                    if not isinstance(top_points, list) or len(top_points) not in (4,):
                        raise ValueError(f"{tid} 必须有四个坐标点")
                    clean_top = []
                    for point in top_points:
                        x, y = float(point[0]), float(point[1])
                        if not (0 <= x <= sample["width"] and 0 <= y <= sample["height"]):
                            raise ValueError(f"{tid} 坐标越界")
                        clean_top.append([round(x, 2), round(y, 2)])
                    lower = [lower_left_point((point[0], point[1]), sample["height"]) for point in clean_top]
                    clean_item = {
                        "tid": str(tid),
                        "content": content,
                        "points": lower,
                        "points_top_left": clean_top,
                        "location": lower_left_bbox(lower, sample["height"]),
                    }
                    if item.get("source") is not None:
                        clean_item["source"] = str(item["source"])
                    if item.get("score") is not None:
                        clean_item["score"] = item["score"]
                    texts[str(tid)] = clean_item
                payload = empty_payload(sample)
                payload["texts"] = texts
                if body.get("preannotation") is not None:
                    payload["preannotation"] = body["preannotation"]
                elif existing.get("preannotation") is not None:
                    payload["preannotation"] = existing["preannotation"]
                annotation_path(output, case_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if body.get("rebuild_training_files", False):
                    writer = OCRAnnotator.__new__(OCRAnnotator)
                    writer.output = output
                    writer.rebuild_training_files()
                self.send_json(payload)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"浏览器标注服务已启动：http://127.0.0.1:{args.port}")
    print(f"服务器远程访问请使用端口转发，当前样本数：{len(samples)}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n标注服务已停止")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    samples = collect_samples(source, args.include_validation)
    if args.check_only:
        print(f"source={source}")
        print(f"samples={len(samples)}")
        print("case_ids=" + ",".join(item["case_id"] for item in samples))
        return
    if args.web or not os.environ.get("DISPLAY"):
        run_web_server(args, samples)
    else:
        OCRAnnotator(args, samples).run()


if __name__ == "__main__":
    main()
