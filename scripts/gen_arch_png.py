"""Generate a PNG version of the system architecture diagram using Pillow.

Run:
    python scripts/gen_arch_png.py

This will create docs/system_architecture.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 300
BG = (255, 255, 255)
OUT = Path(__file__).resolve().parent.parent / "docs" / "system_architecture.png"

def load_font(size: int):
    # Fallback to a default PIL bitmap font if truetype not found
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def center(draw: ImageDraw.ImageDraw, xy, w, h, text, font, fill=(0,0,0)):
    tw, th = draw.textbbox((0,0), text, font=font)[2:]
    x, y = xy
    draw.text((x + (w - tw)/2, y + (h - th)/2), text, font=font, fill=fill, align="center")

def main():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)
    title_font = load_font(28)
    text_font = load_font(18)

    # Layout rectangles
    boxes = [
        (40, 100, 190, 200, ["Camera", "Pcam 5C"]),
        (270, 60, 570, 240, ["FPGA (PL)", "Capture MIPI", "MotionGate (WIP)", "Resize + Letterbox"]),
        (650, 60, 950, 240, ["PS (ARM Cortex-A9)", "YOLOv8n INT8", "NMS", "Abandon logic"]),
        (1000, 100, 1160, 200, ["Outputs", "Alerts / GUI", "Logs"]),
    ]

    for (x1, y1, x2, y2, lines) in boxes:
        d.rectangle([x1, y1, x2, y2], outline=(0,0,0), width=3, fill=(238,240,255))
        for idx, line in enumerate(lines):
            font = title_font if idx == 0 else text_font
            tw, th = d.textbbox((0,0), line, font=font)[2:]
            total_h = sum(d.textbbox((0,0), l, font=(title_font if i==0 else text_font))[3] for i,l in enumerate(lines)) + 8*(len(lines)-1)
            start_y = y1 + ( (y2 - y1) - total_h )/2
            y_line = start_y + idx*(th + 8)
            d.text((x1 + (x2 - x1 - tw)/2, y_line), line, font=font, fill=(0,0,0))

    # Arrows
    def arrow(x1,y1,x2,y2):
        d.line((x1,y1,x2,y2), fill=(0,0,0), width=4)
        # simple triangular head
        head = [(x2,y2), (x2-12,y2-7), (x2-12,y2+7)]
        d.polygon(head, fill=(0,0,0))
    arrow(190,150,270,150)
    arrow(570,150,650,150)
    arrow(950,150,1000,150)

    caption = "Tomo Dataflow: Camera -> PL Pre-processing -> INT8 Inference & Logic -> Outputs"
    tw, th = d.textbbox((0,0), caption, font=text_font)[2:]
    d.text(((WIDTH - tw)/2, HEIGHT - th - 10), caption, font=text_font, fill=(0,0,0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"Saved {OUT}")

if __name__ == "__main__":
    main()
