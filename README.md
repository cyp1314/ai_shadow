# AI Shadow · 旅游人物照姿态阴影生成

把旅游照片中的人物抠出，抹除原图中的人物得到干净背景，再以**半透明姿态阴影**叠回同一场景——用于"模仿别人姿态拍照"的参照图生成。

## 功能

- **人物抠图**：rembg（u2net_human_seg 模型），alpha 蒙版像素级保留人物细节（手指、腿缝不丢失）
- **背景抹除**：OpenCV inpaint 自动填补人物区域，得到与原图同场景的干净背景
- **姿态阴影**：人物蒙版染成半透明黑（不透明度可调），叠合输出最终 PNG
- **两种使用方式**：本地批量脚本 + HTTP 接口

## 环境准备

- Python 3.10 ~ 3.11

```bash
pip install -r requirements.txt
```

> 注意：`mediapipe` 必须 <0.11（1.0 起移除了本项目依赖的 solutions API）。
> 首次运行会自动下载抠图模型（约 170MB）到 `~/.u2net/`。

## 本地批量脚本

```bash
# 默认输出 shadow：原场景抹除人物 + 姿态阴影
python batch_lineart.py 照片目录 -o output --clean-bg

# 把人物姿态阴影叠到另一张背景图上（现场空景照）
python batch_lineart.py 参考人物照.jpg -o output --background 现场背景照.jpg

# 其他样式：虚线轮廓 / 纯剪影 / 剪影+骨架
python batch_lineart.py 照片目录 -o output --styles overlay,silhouette,lineart
```

常用参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model` | rembg 模型（`u2net_human_seg` / `isnet-general-use` 等） | u2net_human_seg |
| `--shadow-alpha` | 阴影不透明度 0~1 | 0.5 |
| `--background` | 指定背景图路径 | 无 |
| `--clean-bg` | 用 inpaint 抹除原图人物作背景 | 关 |
| `--styles` | overlay / silhouette / lineart / shadow（逗号分隔） | shadow |
| `--epsilon` | 轮廓简化强度（仅非 shadow 样式生效），0 为完全保留原形 | 0.0005 |
| `--smooth` | 轮廓 Chaikin 圆角平滑迭代次数 | 3 |
| `--max-side` | 处理前图片长边上限 | 1536 |
| `--no-pose` | 跳过姿态骨架检测 | 关 |

## HTTP 接口

```bash
python api_server.py        # 监听 0.0.0.0:8000，文档见 http://localhost:8000/docs
```

**POST `/api/pose-shadow`**

- 入参：`multipart/form-data`，字段 `file` = 原图（≤20MB）
- 出参：一张 PNG（干净背景 + 姿态阴影），`Content-Type: image/png`
- 结果同时留存在 `uploads_result/时间戳_uuid.png`，路径见响应头 `X-Result-Path`

```bash
curl -o result.png -F "file=@照片.jpg" http://localhost:8000/api/pose-shadow
```

**GET `/health`** — 服务与模型加载状态。CPU 下单张处理约 1~2 秒。

## 项目结构

```
batch_lineart.py    本地批量处理脚本（核心管线，被 API 复用）
api_server.py       FastAPI 接口服务
requirements.txt    依赖清单
uploads_result/     接口结果留存目录（不入库）
output/             批量脚本输出目录（不入库）
```

## 已知局限与后续方向

- inpaint 采用传统算法，大面积人物抹除会发糊；生产环境建议接入 IOPaint（LaMa）或生成式修复 API
- 阴影与背景的叠合按坐标等比缩放，不同机位需用户在 App 内手动拖动/缩放对齐
- 细长物体（登山杖等）可能被抠图模型遗漏，可尝试 `--model isnet-general-use`
