#!/usr/bin/env python3
"""Render per-step and cumulative-best training scores from a SLURM log."""

from __future__ import annotations

import argparse
import re
import struct
import zlib
from pathlib import Path


SCORE_RE = re.compile(r"step:(\d+).*?critic/score/max:([0-9.eE+-]+)")


def read_scores(log_path: Path) -> list[tuple[int, float]]:
    scores: list[tuple[int, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = SCORE_RE.search(line)
        if match:
            scores.append((int(match.group(1)), float(match.group(2))))
    return scores


def points(values: list[tuple[int, float]], *, left: float, top: float, width: float, height: float, low: float, high: float) -> str:
    max_step = max(step for step, _ in values)
    return " ".join(
        f"{left + (step - 1) / max(1, max_step - 1) * width:.1f},"
        f"{top + height - (score - low) / (high - low) * height:.1f}"
        for step, score in values
    )


def render_svg(scores: list[tuple[int, float]], output_path: Path) -> None:
    if not scores:
        raise ValueError("No completed training-step scores found in log")

    cumulative: list[tuple[int, float]] = []
    best = float("-inf")
    for step, score in scores:
        best = max(best, score)
        cumulative.append((step, best))

    width, height = 1200, 720
    left, right, top, bottom = 100, 50, 85, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    high = max(score for _, score in cumulative)
    low = min(0.0, min(score for _, score in scores))
    padding = max(2.0, (high - low) * 0.05)
    high += padding
    max_step = scores[-1][0]

    grid = []
    for i in range(6):
        value = low + (high - low) * i / 5
        y = top + plot_height - (value - low) / (high - low) * plot_height
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 14}" y="{y + 5:.1f}" class="axis-label" text-anchor="end">{value:.1f}</text>'
        )
    for step in range(1, max_step + 1):
        x = left + (step - 1) / max(1, max_step - 1) * plot_width
        grid.append(f'<text x="{x:.1f}" y="{top + plot_height + 28}" class="axis-label" text-anchor="middle">{step}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 26px sans-serif; fill: #172033; }}
    .subtitle {{ font: 15px sans-serif; fill: #56627a; }}
    .axis {{ stroke: #63708a; stroke-width: 1.5; }}
    .grid {{ stroke: #dce2ec; stroke-width: 1; }}
    .axis-label {{ font: 13px sans-serif; fill: #56627a; }}
    .legend {{ font: 15px sans-serif; fill: #26334d; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="38" class="title">Top-k guidance training score history</text>
  <text x="{left}" y="63" class="subtitle">Completed steps 1–{max_step}; critic/score/max</text>
  {''.join(grid)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
  <polyline points="{points(scores, left=left, top=top, width=plot_width, height=plot_height, low=low, high=high)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{points(cumulative, left=left, top=top, width=plot_width, height=plot_height, low=low, high=high)}" fill="none" stroke="#e76f51" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{left + 10}" cy="{height - 40}" r="5" fill="#2563eb"/><text x="{left + 22}" y="{height - 34}" class="legend">Per-step maximum</text>
  <circle cx="{left + 210}" cy="{height - 40}" r="5" fill="#e76f51"/><text x="{left + 222}" y="{height - 34}" class="legend">Historical maximum</text>
  <text x="{left + plot_width}" y="{height - 34}" class="legend" text-anchor="end">Current best: {cumulative[-1][1]:.4f}</text>
</svg>
'''
    output_path.write_text(svg)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def render_png(scores: list[tuple[int, float]], output_path: Path) -> None:
    """Render a dependency-free raster version of the score chart."""
    if not scores:
        raise ValueError("No completed training-step scores found in log")

    cumulative: list[tuple[int, float]] = []
    best = float("-inf")
    for step, score in scores:
        best = max(best, score)
        cumulative.append((step, best))

    width, height = 1200, 720
    left, right, top, bottom = 100, 50, 85, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    low = min(0.0, min(score for _, score in scores))
    high = max(score for _, score in cumulative)
    high += max(2.0, (high - low) * 0.05)
    max_step = scores[-1][0]
    pixels = bytearray([255]) * (width * height * 3)

    def pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            index = (y * width + x) * 3
            pixels[index : index + 3] = bytes(color)

    def line(x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int], thickness: int = 1) -> None:
        steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0))))
        radius = thickness // 2
        for step in range(steps + 1):
            x = round(x0 + (x1 - x0) * step / steps)
            y = round(y0 + (y1 - y0) * step / steps)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    pixel(x + dx, y + dy, color)

    def to_xy(step: int, score: float) -> tuple[float, float]:
        return (
            left + (step - 1) / max(1, max_step - 1) * plot_width,
            top + plot_height - (score - low) / (high - low) * plot_height,
        )

    grid_color, axis_color = (220, 226, 236), (99, 112, 138)
    for i in range(6):
        y = top + plot_height * i / 5
        line(left, y, left + plot_width, y, grid_color)
    line(left, top, left, top + plot_height, axis_color, 2)
    line(left, top + plot_height, left + plot_width, top + plot_height, axis_color, 2)
    for values, color in ((scores, (37, 99, 235)), (cumulative, (231, 111, 81))):
        for (step0, score0), (step1, score1) in zip(values, values[1:]):
            line(*to_xy(step0, score0), *to_xy(step1, score1), color, 3)
        for step, score in values:
            x, y = to_xy(step, score)
            line(x - 3, y, x + 3, y, color, 3)
            line(x, y - 3, x, y + 3, color, 3)

    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3]) for y in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"tEXt", f"Title\x00Top-k guidance score history through step {max_step}".encode())
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    output_path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scores = read_scores(args.log)
    if args.output.suffix.lower() == ".png":
        render_png(scores, args.output)
    else:
        render_svg(scores, args.output)


if __name__ == "__main__":
    main()
