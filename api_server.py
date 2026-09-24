# -*- coding: utf-8 -*-
"""姿态阴影 HTTP 接口

POST /api/pose-shadow   multipart 字段 file=原图
    -> 返回 PNG：原场景抹除人物后，把人物姿态以半透明阴影叠回的效果
    -> 结果同时保存到 uploads_result/ 目录

启动:
    python api_server.py            # 监听 0.0.0.0:8000
接口文档: http://localhost:8000/docs
"""

import io
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from batch_lineart import cutout_person

SAVE_DIR = Path("uploads_result")
MAX_SIDE = 1536          # 处理长边上限
SHADOW_ALPHA = 0.5       # 阴影不透明度
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Pose Shadow API")
_session = None


@app.on_event("startup")
def load_model():
    global _session
    from rembg import new_session

    _session = new_session("u2net_human_seg")


def build_pose_shadow(img: Image.Image) -> bytes:
    rgba = cutout_person(img, _session)

    # 阴影层：人物 alpha 蒙版染成半透明黑
    arr = np.array(rgba).copy()
    arr[:, :, :3] = (15, 15, 15)
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * SHADOW_ALPHA).astype(np.uint8)
    shadow = Image.fromarray(arr)

    # 干净背景：inpaint 抹除原图中的人物
    hole = cv2.dilate((np.array(rgba)[:, :, 3] > 127).astype(np.uint8) * 255,
                      np.ones((15, 15), np.uint8))
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    clean_bg = Image.fromarray(
        cv2.cvtColor(cv2.inpaint(bgr, hole, 10, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB))

    canvas = clean_bg.convert("RGBA")
    canvas.alpha_composite(shadow)

    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


@app.post("/api/pose-shadow")
async def pose_shadow(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片超过 20MB 限制")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(422, "无法读取图片文件")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)

    try:
        png = build_pose_shadow(img)
    except Exception as e:
        raise HTTPException(500, f"处理失败: {e}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    (SAVE_DIR / out_name).write_bytes(png)

    return Response(content=png, media_type="image/png",
                    headers={"Content-Disposition": f'inline; filename="{out_name}"',
                             "X-Result-Path": str((SAVE_DIR / out_name).resolve())})


@app.get("/health")
def health():
    return {"ok": _session is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
