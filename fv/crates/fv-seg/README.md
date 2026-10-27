# fv-seg · 模型获取

运行时依赖（均不入库，遵循各自上游许可）：

| 文件 | 放置路径（默认） | 环境变量覆盖 | 来源 |
|---|---|---|---|
| `onnxruntime.dll`（1.22，本机架构） | `<repo>/deps/ort/onnxruntime.dll` | `FV_SEG_ORT_DYLIB` | [microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)（注意 Windows ARM64 与 x64 各取其版） |
| `modnet.onnx`（人像 matting，26MB） | `<repo>/deps/models/modnet.onnx` | `FV_SEG_MODEL` | [MODNet](https://github.com/ZHKKKe/MODNet)（Apache-2.0 ONNX 导出） |
| `u2net.onnx`（通用显著性，177MB） | `<repo>/deps/models/u2net.onnx` | `FV_SEG_U2NET` | [U^2-Net](https://github.com/xuebinqin/U-2-Net)（Apache-2.0 ONNX 导出） |

ort 依赖 crate 为 `ort 2.0.0-rc.10`（load-dynamic）：首次调用前设置 `ORT_DYLIB_PATH`
（fv-seg 会自动转发）。

没有模型/网络也能用：`fvseg --matte <mask.png>` 直接给定掩码（尺寸不符自动 Triangle 缩放）。
