import json
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "赛题六公开数据集" / "200_train_cases" / "0001"
IMAGE_PATH = SAMPLE_DIR / "0001-KiCad.png"
ANNOTATION_PATH = SAMPLE_DIR / "0001-KiCad_target_named.json"
OUTPUT_PATH = SAMPLE_DIR / "0001-KiCad_bbox_overlay.png"

PALETTE = [
    (220, 55, 48),
    (0, 132, 176),
    (0, 143, 91),
    (210, 119, 23),
    (166, 67, 153),
    (62, 99, 198),
    (132, 137, 24),
    (43, 139, 132),
]


def load_font(size):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def wrap_to_width(text, font, draw, max_width):
    lines = []
    current = ""
    for char in text:
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def main():
    source = Image.open(IMAGE_PATH).convert("RGB")
    with ANNOTATION_PATH.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    components = list(annotation["components"].items())
    image_width, image_height = source.size
    invalid = []
    outside = []
    for key, component in components:
        bbox = component.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            invalid.append(key)
        elif bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_width or bbox[3] > image_height:
            outside.append(key)

    panel_width = 1080
    margin = 18
    gap = 24
    columns = 2
    column_width = (panel_width - 2 * margin - gap) // columns
    title_height = 64
    font = load_font(13)
    title_font = load_font(18)
    line_height = 17
    row_padding = 5

    panel_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    legend_columns = [components[i::columns] for i in range(columns)]
    wrapped_columns = []
    for column in legend_columns:
        wrapped_rows = []
        for index, (key, component) in enumerate(column):
            value = component.get("value")
            value_text = "null" if value is None else str(value)
            text = (
                f"#{index * columns + legend_columns.index(column) + 1:02d} "
                f"key:{key} | Name:{component.get('Name')} | "
                f"type:{component.get('type')} | value:{value_text}"
            )
            wrapped_rows.append(wrap_to_width(text, font, panel_draw, column_width - 14))
        wrapped_columns.append(wrapped_rows)

    legend_height = max(
        sum(max(1, len(lines)) * line_height + row_padding for lines in column)
        for column in wrapped_columns
    )
    canvas_height = max(image_height + title_height, title_height + legend_height + 2 * margin)
    canvas = Image.new("RGB", (image_width + panel_width, canvas_height), (248, 249, 250))
    canvas.paste(source, (0, title_height))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 17), "0001 KiCad - component bounding boxes", fill=(25, 35, 45), font=title_font)
    draw.text(
        (image_width + margin, 17),
        f"{len(components)} components | image {image_width} x {image_height} | bbox [xmin, ymin, xmax, ymax]",
        fill=(25, 35, 45),
        font=font,
    )
    draw.line((image_width, 0, image_width, canvas_height), fill=(205, 210, 215), width=1)

    for index, (key, component) in enumerate(components, start=1):
        x1, y1, x2, y2 = [int(round(value)) for value in component["bbox"]]
        kind = str(component.get("type", ""))
        color = PALETTE[zlib.crc32(kind.encode("utf-8")) % len(PALETTE)]
        box = (x1, y1 + title_height, x2, y2 + title_height)
        draw.rectangle(box, outline=color, width=2)
        tag = f"{index:02d}"
        tag_box = draw.textbbox((0, 0), tag, font=font)
        tag_width = tag_box[2] - tag_box[0] + 8
        tag_height = tag_box[3] - tag_box[1] + 4
        tag_x = min(max(0, x1), image_width - tag_width)
        tag_y = max(title_height, y1 + title_height - tag_height)
        draw.rectangle((tag_x, tag_y, tag_x + tag_width, tag_y + tag_height), fill=color)
        draw.text((tag_x + 4, tag_y + 1), tag, fill=(255, 255, 255), font=font)

    for column_index, column in enumerate(wrapped_columns):
        x = image_width + margin + column_index * (column_width + gap)
        y = title_height + margin
        for lines in column:
            for line in lines:
                draw.text((x, y), line, fill=(35, 43, 50), font=font)
                y += line_height
            y += row_padding

    canvas.save(OUTPUT_PATH)
    print(f"source: {IMAGE_PATH}")
    print(f"annotation: {ANNOTATION_PATH}")
    print(f"output: {OUTPUT_PATH}")
    print(f"image size: {image_width} x {image_height}; components: {len(components)}")
    print(f"invalid boxes: {len(invalid)}; boxes outside image: {len(outside)}")
    if invalid:
        print("invalid keys:", ", ".join(invalid))
    if outside:
        print("outside keys:", ", ".join(outside))


if __name__ == "__main__":
    main()
