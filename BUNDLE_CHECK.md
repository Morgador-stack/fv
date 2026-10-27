# bundle-5090d 交付前终检报告

- 检查日期：2026-10-27
- 检查对象：`D:\WorkSpace\ZCWorkspace\forza-vinyl\bundle-5090d\`
- 检查方式：只读（文件清单比对 / python 标准库解析 JSON 与 PE 头 / 二进制 strings 提取）。未运行任何 exe，未触碰 GPU。
- 结论总览：**二进制与数据本身健康（PE 架构、JSON 内容、包内布局全部过关），但 README 文案滞后于最后一次重新打包（10-27 00:15 的 exe/designs），存在 1 个幽灵文件条目和 1 处过时的 onnxruntime 指引**。

---

## 一、逐项核对表

### 1. 文件完整性 — FAIL（README 侧）

README 声称 → 实际存在情况：

| README 声称 | 实际 | 结果 |
|---|---|---|
| fv-inject.exe | 存在，835,584 B | OK |
| fvfit-gpu.exe | 存在，13,443,072 B | OK |
| fvseg.exe | 存在，12,772,864 B | OK |
| eval_design.exe | 存在，6,385,152 B | OK |
| flatten.exe | 存在，6,306,304 B | OK |
| bench.sh | 存在，2,359 B | OK |
| models/（modnet 26MB + u2net 177MB） | modnet.onnx 25,888,640 B；u2net.onnx 177,357,371 B（均非零，大小与描述一致） | OK |
| designs/（fv3_full / fc_full / fvB_solid300.refined / fvl1_ayy_v2） | **fc_full.json 不存在**；其余 3 个均存在非零 | FAIL |
| INJECT.md | 存在，3,509 B | OK |
| （README 文件清单未列） | **ort/onnxruntime.dll 存在**，21,296,680 B | 漏登记 |

反向核对（目录有、README 没有按名登记）：

| 目录内文件 | README 登记 |
|---|---|
| `ort/onnxruntime.dll` | 未列入文件清单；"注意事项"里只写了去 `<repo>/deps/ort` 拷贝（见问题 P1-2） |
| `designs/fvB_full.json` | 未登记（README 表里写的是 fc_full.json） |
| `designs/fv300softR2.json` | 未登记 |
| `designs/ayylmao.png` | 未登记为文件（仅标题一句"ayylmao 328²"） |
| `.mimosa/`（5 个 json，内部扫描/审查产物） | 未登记，且不应随包交付（见 P2-3） |

### 2. README 时效性 — FAIL（仅 designs 表部分）

| 核对点 | 结果 |
|---|---|
| fv3_full.json 文件存在 | OK（670,437 B） |
| fv3_full.json = 36.11dB | **未复核**：需跑 eval_design.exe（GPU 纪律禁止），只能核对层数 |
| 旧 JSON 未清 | designs/ 内**没有**陈旧残留；问题反过来——README 引用了不存在的 `fc_full.json`（幽灵条目），而实际的 `fvB_full.json`（2177 层）与表中 fc_full 的"2091 层 / 31.16dB"对不上，说明 README 写于 fvB_full 重新生成之前 |
| fvseg 描述含 --model / --tune / --strokes | OK（README L11：`--model modnet/u2net，--gpu，--strokes，--tune`） |
| 提及 flatten.exe | OK（L13） |
| 提及 bench.sh | OK（L14） |

### 3. 设计 JSON 抽验 — PASS（结构层面）

python 标准库解析，全部 5 个 JSON 结构完好（`{type, data:[x,y,w,h,angle], color:[r,g,b,a]}`，无畸形记录）：

| 文件 | shapes 数 | type 分布 | x 范围 | y 范围 | w/h 范围 | 备注 |
|---|---|---|---|---|---|---|
| fv3_full.json | **3000**（=README 声称） | **全部 16** | -0.07~320.77 | 21.43~327.57 | 0.5~109.88 / 0.5~70.86 | PASS |
| fvB_full.json | 2177 | 16×576 + 228×1601 | -0.07~330.55 | 1.11~329.75 | 正常 | 2177 ≠ 表中 fc_full 的 2091 |
| fvB_solid300.refined.json | 300（=README 声称） | 16×300 | 同上 | 同上 | 正常 | x_max=330.55 略超 328 画布（中心坐标+宽度外溢，属正常裁剪语义，仅记录） |
| fvl1_ayy_v2.json | 940（=README 声称） | 16×240 + 228×700 | 0.57~327.87 | 1.16~327.62 | 正常 | |
| fv300softR2.json | 1497 | 16×297 + 228×1200 | 0~326.63 | 0.22~327.11 | 正常 | |

坐标均落在 0~330 预期区间（个别 -0.07 / 330.55 为边界浮点外溢，量级可忽略）；颜色 RGBA 全部 0~255；type 228（矩形）在 INJECT.md 字段对照表中有对应 shape_id 映射，合法。附带确认 `designs/ayylmao.png` 为合法 PNG、328x328，与"ayylmao 328²"描述一致。

### 4. bench.sh 可移植性 — PASS（附必改清单）

- `DIR=$(cd "$(dirname "$0")" && pwd)` + `EXE=$DIR`：完全自洽，五个 exe 就在脚本旁，整包拷走后无需改。
- `IMGS=${IMGS:-/d/WorkSpace/ZCWorkspace/forza-vinyl/refs/forza-painter-fh6/.../Vinyls}`：**默认值是开发机（N1X）绝对路径**。整包拷到别的机器后该路径不存在，12 张图全部 `skip missing`、summary 打出 n=0。好在是 `${IMGS:-默认}` 写法，**设环境变量即可覆盖，不必改脚本**。
- `OUT=$DIR/results/$MODE`：产物写进包内目录，自洽（要求包目录可写）。
- 依赖 bash 环境（Git Bash/MSYS2/WSL）：`date +%s.%N`、`sed -E`、`awk`、`tr` 在 Git Bash 下均可用。
- 隐含脆弱点：从 eval_design 输出用 sed 正则抽 `shapes=` / `PSNR=... dB`，eval 输出格式一变 summary 列就空（脚本不报错，只会打 0 值行）。

**拷包时必须手动处理的路径清单（仅 1 项）**：
1. `IMGS` 环境变量 → 指向目标机上的 FH6 社区涂装语料目录（fromza-painter 内嵌的 `Vinyls` 目录，**语料本身不在包内**）。不设则 bench 全部 skip。

### 5. ort dll 架构 — PASS

python 读 PE 头（e_machine），无 arm64/x64 混装：

| 文件 | machine | 判定 |
|---|---|---|
| fv-inject.exe | 0x8664 | x86_64 |
| fvfit-gpu.exe | 0x8664 | x86_64 |
| fvseg.exe | 0x8664 | x86_64 |
| eval_design.exe | 0x8664 | x86_64 |
| flatten.exe | 0x8664 | x86_64 |
| ort/onnxruntime.dll | 0x8664 | x86_64 |

附加：onnxruntime.dll 内嵌版本串为 **1.22.0**（1.x）。fvseg 二进制内含 ort crate 2.0.0-rc.10 的兼容性校验（要求 GetVersionString 返回 1.x），当前 dll 匹配——**不要在目标机上自行替换成别的 onnxruntime 版本**。另确认 onnxruntime 依赖只存在于 fvseg.exe（fvfit-gpu / eval_design / fv-inject 内无 ORT 相关串）。

### 6. 缺失项清单（下一个工程师视角）— 见第三节逐条

二进制 strings 提取到的 fvseg 实际约定（README 没写全）：
- 环境变量：`FV_SEG_ORT_DYLIB`（dll 路径）、`FV_SEG_MODEL`（modnet 路径）、`FV_SEG_U2NET`（u2net 路径）、`ORT_DYLIB_PATH`（ort crate 底层变量，二选一即可）。
- **内置默认相对路径就是 `ort/onnxruntime.dll` 和 `models/modnet.onnx` / `models/u2net.onnx`**——与包内布局完全一致，即**开箱即用**；`<repo>/deps/...` 是编译期烧进二进制的开发机路径（`D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-seg`），在目标机上必然失效但会被相对默认值兜住。

---

## 二、问题清单（按严重度）

### P1（交付阻断级：文档误导，会直接把人带沟里）

1. **README designs 表引用不存在的 `fc_full.json`**。实际文件是 `fvB_full.json`，2177 层，与表中 fc_full 的 2091 层/31.16dB 均对不上。README（mtime 10-27 00:26）晚于 fvB_full 重新生成（10-27 00:15）却仍留旧条目。下包的人按表找 fc_full.json 会直接找不到。
2. **"注意事项"的 onnxruntime 指引过时且自相矛盾**：README 让用户"放 `deps/ort/`：从 `<repo>/deps/ort` 拷贝 x64 版"。但拿包者没有 repo 可拷；而包内已经自带 `ort/onnxruntime.dll`，且 fvseg 的内置默认相对路径恰好是 `ort/onnxruntime.dll` + `models/*.onnx`，开箱即用。现在的写法反而暗示用户要再造一个 deps/ 目录，或误以为包里缺 dll。env 变量（FV_SEG_ORT_DYLIB / FV_SEG_MODEL / FV_SEG_U2NET）也一字未提。

### P2（应修：信息缺口）

3. **`.mimosa/` 内部工具产物混入交付包**（5 个 json：finding-ledger / history / hook-status / reports）。泄漏开发机内部路径与会话信息，交付前应整目录剔除。
4. **`ort/`、`fvB_full.json`、`fv300softR2.json`、`ayylmao.png` 未在 README 登记**（详见 §1 反向核对表）。
5. **bench 语料不在包内**且 IMGS 默认路径为开发机绝对路径。脚本头与 README 有提示"改 IMGS"，算有交代，但"整包拷走即用"的预期会落空——建议 README 明示"bench 需自备语料并 export IMGS=..."。

### P3（记录在案，不阻断）

6. README 落款"生成于 2026-10-26"，但 exe×3（eval_design/flatten/fvfit-gpu/fvseg）与 designs/ 实际为 10-27 00:15 重新生成——版本对应关系模糊，建议落款改成"二进制与 designs 于 2026-10-27 00:15 重出"。
7. 36.11dB / 31.16dB / 26.97dB / 24.94dB 四个 PSNR 数字本轮**未复核**（GPU 纪律禁跑 exe）。层数与文件存在性已核，数值留给 5090D 实机首跑 bench/eval 时顺带验证。
8. fvB_full / fvB_solid300 的 x_max=330.55 略超 328 画布（中心坐标+半宽外溢，注入时会自然裁剪，无需处理）。
9. README 快速开始没有 fvseg 的示例命令，也没提 `designs/ayylmao.png` 就是现成的测试源图（首跑得自己找图）。

---

## 三、"5 分钟跑通：拟合一张新图 → 注入游戏"指引修正案

以下为**建议替换进 README 的文字**（本轮不动 README 原文，由维护者取舍）。

> ### 快速开始（5 分钟）
>
> 本包开箱自含全部依赖：x64 exe ×5、`ort/onnxruntime.dll`（x64, onnxruntime 1.22.0）、`models/modnet.onnx + u2net.onnx`、测试源图 `designs/ayylmao.png`（328×328）。无需装 CUDA；5090D 走 NVIDIA 驱动自带的 Vulkan 运行时。
>
> ```bash
> # 0. 冒烟测试：用包内现成测试图拟合（GPU，1M 采样）
> fvfit-gpu.exe designs/ayylmao.png --layers 300 --samples 1000000 --opt-res 328 --bg auto --out-prefix out300
>
> # 1. M1.8 精修（+1~1.2dB，2 秒）
> fvfit-gpu.exe --refine-json out300.json designs/ayylmao.png --opt-res 328 --bg auto --diff-iters 300
>
> # 2. 填充软层（3000 预算）
> fvfit-gpu.exe designs/ayylmao.png --resume out300.refined.json --layers 3000 --auto-soft --samples 1000000 --opt-res 328 --bg auto --out-prefix out3000
>
> # 3. 注入（先读 INJECT.md §1 游戏内准备，再三步流程；层数 = 设计层数 + 4 边界层）
> fv-inject.exe --json out3000.json --layer-count 3004 --dry-run
> fv-inject.exe --json out3000.json --layer-count 3004 --yes
> ```
>
> **fvseg（语义分层，可选）**：模型与 dll 的默认查找路径是 exe 旁的 `ort/onnxruntime.dll`、`models/modnet.onnx`、`models/u2net.onnx`——包内布局即默认布局，直接用：
> ```bash
> fvseg.exe designs/ayylmao.png --model u2net --gpu --out-prefix seg300
> ```
> 若把 dll/models 挪了位置，用环境变量指定：`FV_SEG_ORT_DYLIB`（dll 路径）、`FV_SEG_MODEL` / `FV_SEG_U2NET`（模型路径）；没有 dll 时用 `--matte <png>` 绕过模型推理。
>
> **bench（可选）**：12 图基准需要自备 FH 社区涂装语料（不在包内），设 `IMGS` 指向语料 Vinyls 目录后 `./bench.sh`（Git Bash）：
> ```bash
> IMGS=/path/to/Vinyls ./bench.sh            # solid
> IMGS=/path/to/Vinyls ./bench.sh 300 100000 soft
> ```
>
> ### 注意事项（修正后）
>
> - onnxruntime.dll（x64）**包内已自带**于 `ort/`，与 fvseg 默认路径匹配，开箱即用；请勿替换为其他版本（ort crate 2.0.0-rc.10 要求 onnxruntime 1.x，包内为 1.22.0）。N1X 上的 arm64 版与 x64 版不可混用，本包内的是 x64。
> - fvseg 的模型自动发现路径：`models/`（相对 exe）；开发机专用的 `<repo>/deps/...` 路径在目标机无效，如报模型找不到用 `FV_SEG_MODEL` / `FV_SEG_U2NET` 指路，或 `--matte` 绕过。
> - 5090D 需 Vulkan 运行时（NVIDIA 驱动自带）；wgpu 自动选 Vulkan，无需 CUDA。
> - exe 为 MSVC x64 交叉编译产物，目标机需已装 VC++ 2015-2022 Redistributable（x64）（游戏机基本都有，报缺 vcruntime140.dll 时装一下）。
> - 注入风险与流程约束全文适用 INJECT.md；dry-run 抽样不符立即停止。
> - 本包二进制若损坏：在 5090D 上 `cargo build --release -p fv-inject -p fv-gpu --bins`（需 Rust + crates 镜像 rsproxy.cn）。

**交付前建议动作清单**（由维护者执行，本轮未改任何文件）：
1. 按 §二 P1-1 修 README designs 表（fc_full.json → fvB_full.json 2177 层，PSNR 数字待 5090D 复测后回填）。
2. 按 §三 替换"注意事项"中 onnxruntime 段落。
3. 从交付包剔除 `.mimosa/`。
4. （可选）fvB_full / fv300softR2 登记进 designs 表，或删除 fv300softR2.json（若确认无用）。
