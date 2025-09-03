import argparse
import math
import json
import re
import random
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from transformers import pipeline

_DETECTOR = None
_GEN = None

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = pipeline("object-detection", model="facebook/detr-resnet-50")
    return _DETECTOR

def _get_gen():
    global _GEN
    if _GEN is None:
        _GEN = pipeline("text2text-generation", model="google/flan-t5-small")
    return _GEN

def lab_stats(img_array: np.ndarray):
    rgb = np.clip(img_array.astype(np.float32) / 255.0, 0, 1)
    lab = rgb2lab(rgb)
    L = float(lab[..., 0].mean())
    a = float(lab[..., 1].mean())
    b = float(lab[..., 2].mean())
    C = float(math.sqrt(a * a + b * b))
    hue = float((math.degrees(math.atan2(b, a)) % 360.0))
    return L, a, b, C, hue

def average_hex(img_array: np.ndarray) -> str:
    rgb = img_array.reshape(-1, 3).mean(axis=0)
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def guess_undertone(a: float, b: float) -> str:
    return "warm" if b >= a else "cool"

def save_named_palette_preview(pairs, out_path="palette_preview.png"):
    swatch_w, swatch_h = 180, 80
    cols = 2
    rows = (len(pairs) + cols - 1) // cols or 1
    img = Image.new("RGB", (swatch_w * cols, swatch_h * rows), "white")
    draw = ImageDraw.Draw(img)

    for i, (name, hx) in enumerate(pairs):
        r = i // cols
        c = i % cols
        x1, y1 = c * swatch_w, r * swatch_h
        draw.rectangle([x1 + 8, y1 + 8, x1 + swatch_w - 8, y1 + swatch_h - 28], fill=hx)
        label = f"{name} {hx}"
        draw.text((x1 + 12, y1 + swatch_h - 24), label, fill="black")

    img.save(out_path)
    return out_path

def _hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(round(v)))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _name_for_hex(hex_str):
    r, g, b = _hex_to_rgb(hex_str)
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    deg = int(h * 360)
    if s < 0.15 and v > 0.85:  return "Soft White"
    if s < 0.20 and 0.45 < v < 0.85: return "Warm Taupe"
    if s < 0.20 and v <= 0.45: return "Charcoal"
    if   350 <= deg or deg < 15:  return "Crimson"
    elif 15 <= deg < 45:          return "Marigold"
    elif 45 <= deg < 75:          return "Mustard"
    elif 75 <= deg < 150:         return "Emerald"
    elif 150 <= deg < 210:        return "Teal"
    elif 210 <= deg < 255:        return "Azure"
    elif 255 <= deg < 300:        return "Violet"
    else:                         return "Magenta"

def _shift_hsv(hex_str, dh=0.0, ds=0.0, dv=0.0):
    r, g, b = _hex_to_rgb(hex_str)
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h = (h + dh) % 1.0
    s = min(1.0, max(0.0, s + ds))
    v = min(1.0, max(0.0, v + dv))
    R, G, B = [int(round(x*255)) for x in colorsys.hsv_to_rgb(h, s, v)]
    return _rgb_to_hex((R, G, B))

def generate_fallback_colors(skin_hex: str, n: int = 6):
    def deg_to_unit(deg): return (deg % 360) / 360.0

    r, g, b = _hex_to_rgb(skin_hex)
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)

    comps = []
    comps.append(_shift_hsv(skin_hex, dh=0.5))
    comps.append(_shift_hsv(skin_hex, dh=1/3))
    comps.append(_shift_hsv(skin_hex, dh=-1/3))
    comps.append(_shift_hsv(skin_hex, dh=30/360))
    comps.append(_shift_hsv(skin_hex, dh=-30/360))
    comps.append(_shift_hsv(skin_hex, ds=-(s*0.6 + 0.15), dv=0.08))

    seen = set()
    pairs = []
    for hx in comps:
        if hx not in seen:
            seen.add(hx)
            pairs.append((_name_for_hex(hx), hx))

    return pairs[:n]

def crop_person_center(image: Image.Image):
    detector = _get_detector()

    detections = detector(image)
    person_boxes = []
    for d in detections:
        if d.get("label", "").lower() != "person" or d.get("score", 0) <= 0.7:
            continue
        box = d.get("box", {})
        xmin = box.get("xmin", box.get("x1"))
        ymin = box.get("ymin", box.get("y1"))
        xmax = box.get("xmax", box.get("x2"))
        ymax = box.get("ymax", box.get("y2"))
        width = box.get("width")
        height = box.get("height")
        if width is None and None not in (xmin, xmax):
            width = xmax - xmin
        if height is None and None not in (ymin, ymax):
            height = ymax - ymin
        if None in (xmin, ymin, width, height):
            continue
        person_boxes.append({"xmin": xmin, "ymin": ymin, "width": width, "height": height})

    if person_boxes:
        best = max(person_boxes, key=lambda b: b["width"] * b["height"])
        x, y, w, h = best["xmin"], best["ymin"], best["width"], best["height"]
        person = image.crop((x, y, x + w, y + h))
    else:
        W, H = image.size
        side = int(min(W, H) * 0.8)
        left = (W - side) // 2
        top = (H - side) // 2
        person = image.crop((left, top, left + side, top + side))

    W, H = person.size
    cx1, cy1 = int(W * 0.3), int(H * 0.3)
    cx2, cy2 = int(W * 0.7), int(H * 0.7)
    return person.crop((cx1, cy1, cx2, cy2))

_FEWSHOTS = """
Example A:
Input:
L*: 78.2, a*: 12.3, b*: 20.5, Chroma: 23.9, HEX: #EAC7B0
Output JSON:
{"colors":[
  {"name":"Dusty Rose","hex":"#D8A1A6"},
  {"name":"Sage","hex":"#9CB89A"},
  {"name":"Soft Cornflower","hex":"#87A8D0"},
  {"name":"Warm Taupe","hex":"#B2947F"},
  {"name":"Muted Coral","hex":"#E08C7E"},
  {"name":"Pewter","hex":"#8B8F99"}
]}

Example B:
Input:
L*: 48.1, a*: 18.0, b*: 35.7, Chroma: 40.0, HEX: #9A6C4B
Output JSON:
{"colors":[
  {"name":"Teal","hex":"#2E8B8B"},
  {"name":"Deep Forest","hex":"#264E3E"},
  {"name":"Marigold","hex":"#E2A100"},
  {"name":"Berry","hex":"#8C2E5A"},
  {"name":"Azure","hex":"#2C6CC9"},
  {"name":"Warm Sand","hex":"#C4A882"}
]}
""".strip()

def suggest_colors_with_hf_llm(L: float, a: float, b: float, C: float, hex_sample: str, n: int = 6):
    gen = _get_gen()

    rnd_seed = int((abs(hash((round(L,1), round(a,1), round(b,1), round(C,1), hex_sample)))) % 1_000_000)
    random.seed(rnd_seed)

    prompt = f"""
You are a professional color stylist. Recommend {n} flattering clothing colors for the user based on their skin sample.

Guidelines:
- Consider undertone and contrast from CIELAB stats.
- Mix 2 subtle near-neutrals, 2 medium accents, 2 strong statement colors.
- Avoid colors too close to the skin HEX; keep a clear contrast.
- Return STRICT JSON only, following the examples below.

{_FEWSHOTS}

Now respond for this input:
Input:
L*: {L:.1f}, a*: {a:.1f}, b*: {b:.1f}, Chroma: {C:.1f}, HEX: {hex_sample}
Output JSON:
""".strip()

    out = gen(
        prompt,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.08,
        num_return_sequences=1,
    )[0]["generated_text"]

    try:
        json_str = re.search(r"\{.*\}\s*$", out, re.S).group(0)
        data = json.loads(json_str)
        colors = data.get("colors", [])
        cleaned = []
        for c in colors:
            name = str(c.get("name", "")).strip()[:40] or "Color"
            hx = str(c.get("hex", "")).strip()
            if re.fullmatch(r"#?[0-9A-Fa-f]{6}", hx):
                if not hx.startswith("#"):
                    hx = "#" + hx
                cleaned.append((name, hx.upper()))
        seen = set()
        unique = []
        for pair in cleaned:
            if pair[1] not in seen:
                seen.add(pair[1])
                unique.append(pair)

        if unique:
            return unique[:n]
        return generate_fallback_colors(hex_sample, n=n)

    except Exception:
        return generate_fallback_colors(hex_sample, n=n)

def main():
    parser = argparse.ArgumentParser(description="Skintone → HF-suggested colors (no hardcoded palettes).")
    parser.add_argument("--image", required=True, help="Path to face image (jpg/png)")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")

    face_region = crop_person_center(image)

    face_np = np.array(face_region)
    L, a, b, C, hue = lab_stats(face_np)
    skin_hex = average_hex(face_np)
    undertone = guess_undertone(a, b)

    pairs = suggest_colors_with_hf_llm(L, a, b, C, skin_hex, n=6)

    out_img = save_named_palette_preview(pairs, "palette_preview.png")

    print("=== Skintone → HF-Suggested Colors ===")
    print(f"Sample HEX: {skin_hex}  |  Undertone (heuristic): {undertone.upper()}")
    print(f"L*: {L:.1f}, a*: {a:.1f}, b*: {b:.1f}, Chroma: {C:.1f}, Hue: {hue:.0f}°")
    print("Suggested colors:")
    for name, hx in pairs:
        print(f"- {name}: {hx}")
    print(f"Saved swatches to: {out_img}")
    print("Tip: Use a clear, neutral-lit photo for best results.")

if __name__ == "__main__":
    main()
