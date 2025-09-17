#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kanji stroke-order animator for phrases (e.g., 火曜日) using KanjiVG.
- Loads each kanji's SVG, reads stroke <path d="..."> in order.
- Draws strokes progressively to build frames.
- Exports an animated GIF (and optional MP4).

Dependencies:
  pip install pillow svgpathtools numpy imageio imageio-ffmpeg
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import parse_path, Path
import imageio.v2 as imageio

SVG_NS = "{http://www.w3.org/2000/svg}"
KANJIVG_VIEWBOX = (0.0, 0.0, 109.0, 109.0)  # canonical KanjiVG coordinate space


# ----------------------------- SVG / KanjiVG -----------------------------

def hex_codepoint_5(ch: str) -> str:
    """Lowercase zero-padded 5-hex codepoint used by KanjiVG filenames."""
    return f"{ord(ch):05x}"

def kanjivg_file_for_char(ch: str, kanjivg_dir: str) -> str:
    """Return file path like '<dir>/065e5.svg' for '日'."""
    return os.path.join(kanjivg_dir, f"{hex_codepoint_5(ch)}.svg")

def load_kanji_strokes(kanji_svg_path: str) -> List[Path]:
    """
    Read a KanjiVG SVG and return a list of svgpathtools Path objects in stroke order.
    KanjiVG stores strokes as <path> elements, ordered by stroke order.
    """
    if not os.path.exists(kanji_svg_path):
        raise FileNotFoundError(f"KanjiVG file not found: {kanji_svg_path}")

    tree = ET.parse(kanji_svg_path)
    root = tree.getroot()

    # Gather all <path> in document order.
    d_strings = []
    for elem in root.iter():
        if elem.tag == SVG_NS + "path" and "d" in elem.attrib:
            d_strings.append(elem.attrib["d"])

    if not d_strings:
        raise ValueError(f"No path strokes found in {kanji_svg_path}")

    strokes = []
    for d in d_strings:
        try:
            strokes.append(parse_path(d))
        except Exception as e:
            raise ValueError(f"Failed to parse a stroke path in {kanji_svg_path}: {e}")

    return strokes


# ----------------------------- Geometry utils -----------------------------

def transform_path(p: Path, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> Path:
    """Scale (uniform) and translate a path: (x, y) -> (scale*x + dx, scale*y + dy)."""
    # svgpathtools Path supports complex-affine via scaled()/translated()
    return p.scaled(scale) .translated(dx + 1j * dy)

def sample_path_by_length(p: Path, length_limit: float, n_samples: int) -> List[complex]:
    """
    Sample points from start of path up to 'length_limit' (clamped to [0, p.length()])
    using arclength parametrization (ilength). Returns complex points.
    """
    L = p.length()
    if L == 0:
        return []
    s_max = max(0.0, min(length_limit, L))
    if s_max == 0:
        return []
    ss = np.linspace(0.0, s_max, max(2, n_samples))
    pts = [p.point(p.ilength(s)) for s in ss]
    return pts

def draw_polyline(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], width: int, color=(20, 40, 60)):
    """Draw a polyline as consecutive line segments."""
    if len(pts) < 2:
        return
    draw.line(pts, fill=color, width=width, joint="curve")


# ----------------------------- Rendering -----------------------------

def render_frame(
    canvas_w: int, canvas_h: int,
    painted_strokes: List[List[Tuple[float, float]]],
    partial_stroke: List[Tuple[float, float]] = None,
    bg_color=(250, 252, 255),
    grid=True,
    grid_color=(210, 220, 235),
    stroke_color=(18, 38, 60),
    stroke_width=10,
) -> Image.Image:
    """Compose a single frame as PIL Image from full strokes + optional partial stroke."""
    img = Image.new("RGBA", (canvas_w, canvas_h), bg_color)
    drw = ImageDraw.Draw(img)

    # Optional light square grids behind each character
    if grid:
        # Try to detect cell size by counting vertical separators via stroke bounds? Simpler: ignore.
        pass

    # Draw fully painted strokes
    for poly in painted_strokes:
        draw_polyline(drw, poly, width=stroke_width, color=stroke_color)

    # Draw the current growing stroke on top (slightly darker)
    if partial_stroke and len(partial_stroke) >= 2:
        draw_polyline(drw, partial_stroke, width=stroke_width, color=(10, 25, 45))

    return img


def path_to_polyline(p: Path, samples: int) -> List[Tuple[float, float]]:
    """Convert a full path to a list of (x, y) points."""
    if p.length() == 0:
        return []
    ts = np.linspace(0.0, 1.0, max(2, samples))
    pts = [p.point(t) for t in ts]
    return [(pt.real, pt.imag) for pt in pts]


# ----------------------------- Animator -----------------------------

def build_phrase_strokes(
    phrase: str,
    kanjivg_dir: str,
    char_size: int,
    gap_px: int,
    padding: int,
) -> Tuple[List[Path], int, int]:
    """
    Load and spatially arrange all strokes for the phrase (left-to-right),
    returning (all_strokes, canvas_w, canvas_h).
    Each character's strokes are scaled to fit char_size x char_size and shifted by gap.
    """
    x0, y0, w, h = KANJIVG_VIEWBOX
    scale = char_size / w

    all_strokes: List[Path] = []
    n = len(phrase)

    for i, ch in enumerate(phrase):
        svg_path = kanjivg_file_for_char(ch, kanjivg_dir)
        strokes = load_kanji_strokes(svg_path)
        # Place this character's strokes in its cell
        dx = padding + i * (char_size + gap_px)
        dy = padding
        for p in strokes:
            all_strokes.append(transform_path(p, scale=scale, dx=dx, dy=dy))

    canvas_w = padding*2 + n*char_size + (n-1)*gap_px
    canvas_h = padding*2 + char_size
    return all_strokes, canvas_w, canvas_h


def animate_phrase(
    phrase: str,
    kanjivg_dir: str,
    out_gif: str,
    out_mp4: str = None,
    char_size: int = 256,
    gap_px: int = 40,
    padding: int = 32,
    stroke_width: int = 10,
    fps: int = 20,
    ms_per_stroke: int = 700,
    ms_gap: int = 150,
    samples_per_stroke: int = 220,
) -> None:
    """
    Build animation frames across ALL strokes in the phrase (stroke order within each char is kept;
    global order is char1 then char2 ...). Saves GIF and optional MP4.
    """
    # Prepare strokes and canvas
    strokes, W, H = build_phrase_strokes(
        phrase, kanjivg_dir, char_size, gap_px, padding
    )
    # Convert timing to frame counts
    frame_time_ms = 1000 / fps
    frames_per_stroke = max(1, int(round(ms_per_stroke / frame_time_ms)))
    frames_gap = int(round(ms_gap / frame_time_ms))

    frames: List[Image.Image] = []
    painted_polylines: List[List[Tuple[float, float]]] = []

    for idx, p in enumerate(strokes):
        L = p.length()
        if L == 0:
            # Edge case: empty stroke
            painted_polylines.append([])
            # keep at least one frame gap to show "no-op"
            frames.append(render_frame(W, H, painted_polylines, None,
                                       stroke_width=stroke_width))
            continue

        # Progressive drawing for this stroke
        for f in range(frames_per_stroke):
            frac = (f + 1) / frames_per_stroke
            s_lim = L * frac
            pts_complex = sample_path_by_length(p, s_lim, n_samples=max(8, samples_per_stroke//2))
            partial = [(z.real, z.imag) for z in pts_complex]
            img = render_frame(W, H,
                               painted_polylines,
                               partial_stroke=partial,
                               stroke_width=stroke_width)
            frames.append(img)

        # After finishing, bake this stroke into painted set
        full_poly = path_to_polyline(p, samples=max(12, samples_per_stroke))
        painted_polylines.append(full_poly)

        # Add gap frames (pause)
        for _ in range(frames_gap):
            frames.append(render_frame(W, H, painted_polylines, None, stroke_width=stroke_width))

    # --- Save GIF robustly (palette/size consistent) ---
    # Ensure all frames are same size/mode and quantized adaptively for GIF
    qframes = []
    for im in frames:
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        qframes.append(im.convert("P", palette=Image.ADAPTIVE, colors=255))

    duration = int(round(1000 / fps))  # per-frame duration ms
    qframes[0].save(
        out_gif,
        save_all=True,
        append_images=qframes[1:],
        loop=0,
        duration=duration,
        optimize=False,
        disposal=2,
    )
    print(f"[OK] GIF saved: {out_gif}  ({len(qframes)} frames @ {fps} FPS)")

    # --- Optional MP4 (H.264) ---
    if out_mp4:
        # Convert RGBA -> RGB for video encoder
        rgb_frames = [im.convert("RGB") for im in frames]
        imageio.mimsave(out_mp4, rgb_frames, fps=fps, quality=9)
        print(f"[OK] MP4 saved: {out_mp4}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Animate Japanese kanji phrase using KanjiVG.")
    ap.add_argument("phrase", type=str, help="Phrase (e.g., 火曜日)")
    ap.add_argument("--kanjivg", type=str, required=True, help="Path to KanjiVG 'kanji' directory (contains xxxxx.svg)")
    ap.add_argument("--out", type=str, default="strokes.gif", help="Output GIF path")
    ap.add_argument("--mp4", type=str, default=None, help="Optional MP4 output path")

    ap.add_argument("--char-size", type=int, default=256, help="Size of each character cell in pixels")
    ap.add_argument("--gap", type=int, default=40, help="Gap between characters in pixels")
    ap.add_argument("--padding", type=int, default=32, help="Canvas padding")

    ap.add_argument("--stroke-width", type=int, default=10, help="Stroke width in pixels")
    ap.add_argument("--fps", type=int, default=20, help="Frames per second")
    ap.add_argument("--ms-per-stroke", type=int, default=700, help="Milliseconds to draw each stroke")
    ap.add_argument("--ms-gap", type=int, default=150, help="Pause after each stroke (ms)")
    ap.add_argument("--samples-per-stroke", type=int, default=220, help="Polyline samples per stroke (quality vs speed)")

    args = ap.parse_args()

    animate_phrase(
        phrase=args.phrase,
        kanjivg_dir=args.kanjivg,
        out_gif=args.out,
        out_mp4=args.mp4,
        char_size=args.char_size,
        gap_px=args.gap,
        padding=args.padding,
        stroke_width=args.stroke_width,
        fps=args.fps,
        ms_per_stroke=args.ms_per_stroke,
        ms_gap=args.ms_gap,
        samples_per_stroke=args.samples_per_stroke,
    )

if __name__ == "__main__":
    main()
