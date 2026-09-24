# -*- coding: utf-8 -*-
"""本地批量处理人物照片：抠图 -> 轮廓提取 -> 姿态骨架 -> 输出线稿 (SVG + PNG)

样式(--styles):
    shadow      原图背景 + 半透明人物阴影剪影（姿态参照用）  [默认]
    overlay     原图背景 + 人物虚线轮廓(+骨架)
    silhouette  黑色剪影
    lineart     剪影 + 骨架

用法:
    python batch_lineart.py 照片目录 -o 输出目录
    python batch_lineart.py 照片目录 --shadow-alpha 0.4
    python batch_lineart.py 照片目录 --styles overlay --dash 20 14
"""

import argparse
import base64
import io
import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# MediaPipe Pose 33 关键点的连线（骨骼）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
]
# 只画四肢+躯干主骨架时可关掉面部点
FACE_LANDMARKS = set(range(0, 11))


def cutout_person(img: Image.Image, session) -> Image.Image:
    """rembg 抠图，返回 RGBA（透明背景）"""
    from rembg import remove

    return remove(img, session=session, only_mask=False)


def chaikin_smooth(pts: np.ndarray, iterations: int) -> np.ndarray:
    """闭合折线圆角平滑（切角法），消除像素锯齿"""
    pts = pts.astype(np.float64)
    for _ in range(iterations):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, axis=0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, axis=0)
        pts = np.empty((len(pts) * 2, 2), np.float64)
        pts[0::2], pts[1::2] = q, r
    return np.round(pts).astype(np.int32)


def alpha_to_contours(alpha: np.ndarray, epsilon_ratio: float, min_area: float,
                      smooth_iters: int = 3):
    """透明通道 -> 高斯平滑 -> findContours -> 轻微简化 -> 圆角平滑
    返回 [np.array([[x,y],...])]"""
    k = max(3, int(min(alpha.shape) * 0.01) | 1)  # 核大小随图片尺寸
    blurred = cv2.GaussianBlur(alpha, (k, k), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if epsilon_ratio > 0:  # 仅做极轻简化控制点数，默认几乎保留原形
            eps = epsilon_ratio * cv2.arcLength(c, closed=True)
            c = cv2.approxPolyDP(c, eps, closed=True)
        pts = chaikin_smooth(c.reshape(-1, 2), smooth_iters)
        polys.append(pts)
    polys.sort(key=lambda p: cv2.contourArea(p.astype(np.float32)), reverse=True)
    return polys


def detect_pose(img_rgb: np.ndarray, include_face: bool):
    """MediaPipe 姿态估计，返回 (像素坐标列表, 连线列表)；无人时返回 ([], [])"""
    import mediapipe as mp

    h, w = img_rgb.shape[:2]
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        # mediapipe 需要 RGB
        result = pose.process(img_rgb)
    if not result.pose_landmarks:
        return [], []
    lms = result.pose_landmarks.landmark
    points = [(lm.x * w, lm.y * h, lm.visibility) for lm in lms]
    edges = []
    for a, b in POSE_CONNECTIONS:
        if not include_face and (a in FACE_LANDMARKS or b in FACE_LANDMARKS):
            continue
        pa, pb = points[a], points[b]
        if min(pa[2], pb[2]) < 0.5:
            continue
        edges.append(((pa[0], pa[1]), (pb[0], pb[1])))
    return points, edges


def polys_to_svg_paths(polys) -> str:
    d_parts = []
    for p in polys:
        seq = " ".join(f"{x},{y}" for x, y in p)
        first, rest = seq.split(" ", 1)
        d_parts.append(f"M {first} L {rest} Z")
    return " ".join(d_parts)


def draw_dashed_polyline(draw, pts, dash, gap, fill, width):
    """沿闭合折线画虚线（PIL 无原生虚线，手动按长度切段）"""
    pts = [tuple(map(float, p)) for p in pts]
    pts.append(pts[0])
    on, remaining = True, dash
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg == 0:
            continue
        ux, uy = (x2 - x1) / seg, (y2 - y1) / seg
        pos = 0.0
        while pos < seg:
            take = min(remaining, seg - pos)
            if on:
                draw.line([(x1 + ux * pos, y1 + uy * pos),
                           (x1 + ux * (pos + take), y1 + uy * (pos + take))],
                          fill=fill, width=width)
            pos += take
            remaining -= take
            if remaining <= 0:
                on = not on
                remaining = dash if on else gap


def parse_color(s: str):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def img_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    out = img if fmt == "PNG" else img.convert("RGB")
    out.save(buf, fmt, quality=88) if fmt == "JPEG" else out.save(buf, "PNG")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def build_svg(w, h, polys, edges, style: str, bg: Image.Image, args,
              shadow_img: Image.Image = None) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}">']
    d = polys_to_svg_paths(polys)
    color = args.outline_color
    if style in ("overlay", "shadow"):
        parts.append(f'<image href="{img_to_data_uri(bg)}" x="0" y="0" '
                     f'width="{w}" height="{h}" preserveAspectRatio="none"/>')
        if style == "shadow":
            # 直接用抠图蒙版像素级上色，腿缝等细节不丢失
            parts.append(f'<image href="{img_to_data_uri(shadow_img, "PNG")}" '
                         f'x="0" y="0" width="{w}" height="{h}" '
                         f'preserveAspectRatio="none"/>')
        elif d:
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" '
                         f'stroke-width="{args.dash_width}" '
                         f'stroke-dasharray="{args.dash[0]} {args.dash[1]}" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
    else:
        parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')
        if d:
            if style == "outline":
                parts.append(f'<path d="{d}" fill="none" stroke="black" '
                             f'stroke-width="3" stroke-linejoin="round"/>')
            else:
                parts.append(f'<path d="{d}" fill="black"/>')
    if style in ("lineart", "overlay"):
        for (x1, y1), (x2, y2) in edges:
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                         f'y2="{y2:.1f}" stroke="#e74c3c" stroke-width="4" '
                         f'stroke-linecap="round"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_png(w, h, polys, edges, style: str, bg: Image.Image, args,
              shadow_img: Image.Image = None) -> Image.Image:
    if style in ("overlay", "shadow"):
        canvas = bg.convert("RGBA").copy()
    else:
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    color = (*parse_color(args.outline_color), 255)
    if style == "shadow":
        canvas.alpha_composite(shadow_img)  # 蒙版像素级阴影，细节完整
    elif style == "overlay":
        for p in polys:
            draw_dashed_polyline(draw, p, args.dash[0], args.dash[1],
                                 color, args.dash_width)
    elif style != "outline":
        for p in polys:
            draw.polygon([tuple(pt) for pt in p], fill=(20, 20, 20, 255))
    if style in ("lineart", "overlay"):
        for (x1, y1), (x2, y2) in edges:
            draw.line([(x1, y1), (x2, y2)], fill=(231, 76, 60, 255), width=5)
            for cx, cy in ((x1, y1), (x2, y2)):
                r = 6
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(255, 255, 255, 255), outline=(231, 76, 60, 255), width=3)
    return canvas


def process_one(path: Path, out_dir: Path, session, args) -> None:
    img = Image.open(path).convert("RGB")
    if max(img.size) > args.max_side:  # 缩小长边，加快推理
        scale = args.max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    stem = path.stem

    rgba = cutout_person(img, session)
    cut_path = out_dir / f"{stem}_cutout.png"
    rgba.save(cut_path)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    # shadow 样式：直接用抠图 alpha 蒙版上色，与原图轮廓像素级一致
    shadow_img = None
    if "shadow" in styles:
        arr = np.array(rgba).copy()
        arr[:, :, :3] = (15, 15, 15)
        arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * args.shadow_alpha).astype(np.uint8)
        shadow_img = Image.fromarray(arr)

    bg_img = img
    if args.clean_bg:  # 用 inpaint 抹除原图中的人物，得到同场景干净背景
        alpha = np.array(rgba)[:, :, 3]
        hole = cv2.dilate((alpha > 127).astype(np.uint8) * 255,
                          np.ones((15, 15), np.uint8))
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        fixed = cv2.inpaint(bgr, hole, 10, cv2.INPAINT_TELEA)
        bg_img = Image.fromarray(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    elif args.background:  # 把参考图的人物姿态按坐标缩放到用户选定的背景图上
        bg_img = Image.open(args.background).convert("RGB")
        if max(bg_img.size) > args.max_side:
            scale = args.max_side / max(bg_img.size)
            bg_img = bg_img.resize((int(bg_img.width * scale), int(bg_img.height * scale)),
                                   Image.LANCZOS)
        if shadow_img is not None:
            shadow_img = shadow_img.resize(bg_img.size, Image.LANCZOS)

    w, h = bg_img.size
    polys = []
    if {"overlay", "silhouette", "lineart"} & set(styles):
        polys = alpha_to_contours(np.array(rgba)[:, :, 3],
                                  epsilon_ratio=args.epsilon,
                                  min_area=args.min_area_ratio * rgba.width * rgba.height,
                                  smooth_iters=args.smooth)
        if args.background:
            sx, sy = w / rgba.width, h / rgba.height
            polys = [(p * np.array([sx, sy])).astype(np.int32) for p in polys]

    edges = []
    if not args.no_pose and {"overlay", "lineart"} & set(styles):
        points, edges = detect_pose(np.array(img), include_face=False)

    for style in styles:
        (out_dir / f"{stem}_{style}.svg").write_text(
            build_svg(w, h, polys, edges, style, bg_img, args, shadow_img), encoding="utf-8")
        build_png(w, h, polys, edges, style, bg_img, args, shadow_img).save(
            out_dir / f"{stem}_{style}.png")
    print(f"[OK] {path.name}: cutout + {', '.join(styles)} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="批量人物照片 -> 剪影/线稿")
    parser.add_argument("input", help="照片目录（或单个图片文件）")
    parser.add_argument("-o", "--output", default="output", help="输出目录")
    parser.add_argument("--model", default="u2net_human_seg",
                        help="rembg 模型: u2net_human_seg / u2net / isnet-general-use / birefnet-general")
    parser.add_argument("--epsilon", type=float, default=0.0005,
                        help="轮廓简化强度（占周长比例），0=完全保留原图轮廓（默认，已含圆角平滑）；"
                             "调大（如0.004）得到几何折线风格")
    parser.add_argument("--smooth", type=int, default=3,
                        help="轮廓圆角平滑迭代次数，0关闭，默认3")
    parser.add_argument("--min-area-ratio", type=float, default=0.005,
                        help="忽略小于该面积占比的轮廓（去噪点），默认 0.005")
    parser.add_argument("--max-side", type=int, default=1536,
                        help="处理前图片长边上限，默认 1536")
    parser.add_argument("--styles", default="shadow",
                        help="输出样式(逗号分隔): shadow / overlay / silhouette / lineart，默认 shadow")
    parser.add_argument("--shadow-alpha", type=float, default=0.5,
                        help="shadow 样式人物阴影不透明度 0~1，默认 0.5")
    parser.add_argument("--background",
                        help="背景图路径：把参考人物的姿态阴影叠到这张背景上（模仿姿态拍照的参照图）")
    parser.add_argument("--clean-bg", action="store_true",
                        help="用原图作背景，但先 inpaint 抹除其中的人物（同场景演示用）")
    parser.add_argument("--dash", type=int, nargs=2, default=[20, 14],
                        metavar=("实线", "间隔"), help="虚线长度，默认 20 14")
    parser.add_argument("--dash-width", type=int, default=5, help="虚线粗细，默认 5")
    parser.add_argument("--outline-color", default="#e74c3c",
                        help="轮廓虚线颜色，默认 #e74c3c（红色）")
    parser.add_argument("--no-pose", action="store_true", help="跳过姿态骨架")
    args = parser.parse_args()

    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted(
        p for p in in_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        sys.exit(f"未找到图片: {args.input}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 rembg 模型: {args.model}（首次运行会自动下载 ~170MB）")
    from rembg import new_session  # 延迟导入，报依赖缺失更友好
    session = new_session(args.model)

    failed = 0
    for f in files:
        try:
            process_one(f, out_dir, session, args)
        except Exception:
            failed += 1
            print(f"[FAIL] {f.name}")
            traceback.print_exc()
    print(f"\n完成: {len(files) - failed}/{len(files)}，输出在 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
