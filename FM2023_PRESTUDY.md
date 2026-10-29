# FM2023 涂装层表逆向·可行性预研（FM2023_PRESTUDY）

日期：2026-10-28 ｜ 状态：**纯规划文档，零逆向操作**（本文写作期间未运行任何扫描/连接游戏）
依据：STUDY.md §1-2/§5（FH6 方法论与 RE 协议）、NEXT_IDEAS.md §5/M1、
`fv/crates/fv-inject/src/{layout,scan,write_process,mem,win,convert}.rs` 实读、
`refs/ForzaDesigner6/fd6/inject/patterns/fh6_patterns.json` 实读、
2026-10-28 WebSearch 外部事实核查（渠道与反作弊，来源见 §D.0）。

---

## 0. 结论速览

| 问题 | 答案 |
|---|---|
| 方法论可迁移吗 | **~100%**。模板球+内存 diff 协议（STUDY §5）本就是为 FM2023 预写的 |
| 工具壳要改多少 | **~90% 复用**。`layout.rs` 全数据驱动（偏移/shape_id/除数/executables/扫描参数全是 JSON 字段），换游戏=换 JSON；需新写 RE 模式（dump/differ/通用锚点扫描）3-5 天 |
| FM2023 反作弊 | **自研（Turn 10/MS），非 EAC/BattlEye**（两源一致）；社区有 EAC 误传 → 列为 G0 必确认项，确认方式见 §B.9 |
| 社区有没有现成 CE 表 | **未知，G0 之核心问题**。refs 全家族（forza-painter/FD6/vinylizer/FH6-Painter）均只支持 Horizon——FM2023 注入工具先例为零的先验成立，但 CE 表不等于注入工具，须查 |
| 最大风险 | 封号（FM 自研反作弊比 EAC "更严"的社区报告）+ 层表拓扑可能改掉 u64 指针数组 |
| 总成本 | G0 调研 **48h 零成本可先行**；立项主体 1.5-3 周 part-time + 5090D 会话 2-4 次 |
| 建议动作 | **本周做 G0**（§A 协议），按 §E 判据打分后再决定立项 |

---

## §A G0 调研设计（48h，零成本，纯网络）

### A.1 目标

在不动游戏的前提下回答三件事：
1. **Q1（层表资源）**：社区是否已有 FM2023 的 CE 表/内存结构文档（含涂装层表）？
2. **Q2（反作弊实证）**：FM2023 PC 上内存工具的使用先例与封禁证据链是什么样？
3. **Q3（工具生态）**：有没有任何 FM2023 的内存读写工具（trainer/注入器/.dump 工具）可作"可读可写"旁证？

### A.2 渠道清单（按性价比排序）

| # | 渠道 | 查什么 | 优先级 |
|---|---|---|---|
| 1 | **fearlessrevolution.com**（FearLess Cheat Engine 论坛）Tables 板块 | Forza 系列表传统（FH4 表有存档先例）；搜 "Forza Motorsport" | ★★★ |
| 2 | **GitHub 搜索**（code + repos + issues） | `forza motorsport vinyl`、`FM2023 livery memory`、`forzamotorsport.exe`、`LiveryGroup`；重点翻 **ForzaMods/Forza-Mods-AIO**（其 issue 区出现"Anti Cheat Failed to Disable"，证明存在客户端自研反作弊模块）与 forza-painter 全家族（确认其 support matrix 是否真的不含 Motorsport） | ★★★ |
| 3 | **unknowncheats.me**（Other Racing Games 等板块） | 反-cheat 绕过讨论、内存结构帖；UC 帖子偏技术细节，是反作弊机制的最佳信源 | ★★★ |
| 4 | **r/ForzaMotorsport、r/forza、r/cheatengine** | 封禁案例（ban 原因/时长）、CE 使用体验、"cheat table" 讨论帖 | ★★☆ |
| 5 | **Steam 社区 app 2440510 讨论区** | ban 申诉帖、反作弊抱怨（注意区分：FH5 的讨论区 app 是 1551360，别混） | ★★☆ |
| 6 | **Nexus Mods — Forza Motorsport (2023)** 板块 | 有哪些 mod 作者、有无内存类工具作者可私信 | ★★☆ |
| 7 | **trainer 站**（flingtrainer / Cheat Happens 等） | FM2023 trainer 是否存在。**有 trainer ≠ 有层表**，但 trainer 能改内存=反作弊未阻断读写的强旁证（Cheat Happens 2023-10 曾称 FM "反作弊比 EAC 略强"导致无 trainer——此说法本身即证据，去验证现状） | ★★☆ |
| 8 | **中文渠道**：Bilibili（"极限竞速 涂装 修改/批量"）、NGA 极限竞速板、3DM/游侠修改器区、贴吧 | 中文修改器产业历来覆盖 Forza；视频比论坛更容易暴露"用的什么工具" | ★☆☆ |
| 9 | **Discord**：forza-painter 社区、FD6（其 README 自带 Discord 集成）、FM 涂装社区 | 私下问活跃作者"FM 结构有没有人看过"；G0 后期最有效的一步 | ★☆☆ |
| 10 | opencheattables.org 等第三方表站 | **只记录线索，不下载执行**（社区有 bundleware 投诉） | ★☆☆ |

### A.3 搜索关键词

- 英文：`"Forza Motorsport" cheat table`；`"Forza Motorsport 2023" CE table`；
  `Forza Motorsport livery memory hack`；`Forza Motorsport vinyl layer pointer`；
  `Forza Motorsport unlimited layers paint`；`forzamotorsport.exe cheat engine`；
  `site:github.com "forza motorsport" memory`；`site:fearlessrevolution.com forza motorsport`；
  `site:unknowncheats.me forza motorsport`
- 中文：`极限竞速 CE 修改 涂装`；`极限竞速2023 修改器`；`forza 涂装 注入 内存`；
  `极限竞速 风灵月影 修改器`
- 结构关键词（打捞逆向帖）：`LiveryGroup`、`layer table`、`layer_count`、`vinyl group struct`

### A.4 CE 表判别特征：怎么判断一张表"含涂装层表"

强特征（命中即 G0-GO，FH6 同构）：
1. 表内有**u16 层计数条目**：其地址在游戏内建 N 层测试涂装后读出值恰为 N，增删一层值 ±1；
2. 附近（同结构 ±0x100 内）有 **u64 指针**指向私有堆，指向处可解出 N 个 8 字节步长的指针数组，数组元素各自指向独立堆块；
3. 条目命名含 `livery / vinyl / layer / paint group / layer count` 词汇，或表作者附结构图；
4. CE 表以多级指针形式书写（`pGroup+0x5A` 风格）且指针基址可随版本重找（green address 优先）。

弱信号（不算命中，只作反作弊旁证）：credits/coins/level/DRS 类经济与进度表——说明 CE 可附加、内存可读，但不提供涂装结构知识。

验证一张候选表的最小流程（在拿到表后、任何写操作前）：
建 37 层（质数防撞）测试涂装 → 表中找值 37 的 u16 条目 → 游戏内加/删一层复核 ±1 → 读指针数组首项，
dump 其指向块 0x140 字节存档。全程只读。

### A.5 调研协议与产出

- D1 上午：渠道 1/2（每渠道 ≤30min，命中 URL 全记录）；D1 下午：渠道 3/4/5；
  D2 上午：渠道 7/8；D2 下午：渠道 9 私信抛出 + 汇总打分（§E G0 判据）。
- 产出 `G0_FINDINGS.md`：渠道×命中矩阵、证据链接、候选表清单（含判别特征核对）、
  反作弊证据链（每条标注一手/二手）、Q1/Q2/Q3 三答案与 go/no-go 打分。
- 纪律：G0 期间不下载执行任何表文件（恶意软件面）；不安装游戏；不创建账号。

---

## §B FH6 → FM2023 差异清单

说明：FM2023 与 FH6 同属 Forza Tech 世系但为不同工作室分支（T10 vs PG）且隔了一代渲染重构；
以下"预期"基于两代架构常识 + FH6 三方交叉验证经验，**每一项都标注确认方式，未确认前一律按未知处理**。

| # | 维度 | FH6 现状（已验证） | FM2023 预期 | 影响 | 确认方式 |
|---|---|---|---|---|---|
| B.1 | 引擎/编辑器世系 | LiveryGroup+Layer 概念，三工具交叉一致 | 概念世系同源（涂装编辑器存在且分图层），**偏移大概率全变**（FM2023 是 2023 新重构，涂装编辑器 UI 也重做过） | layout 数值全重测（预期内） | G1 扫描 + §C diff |
| B.2 | 可执行名/打包 | `forzahorizon6.exe` 等（layout.executables Vec） | exe 名未确认（预计 `forzamotorsport*.exe` 系）；Steam 与 MS Store/XGP 可能不同二进制 | `executables` 列表天然多名支持；layout 文件按 (版本,平台) 命名 | 安装目录 + 任务管理器进程列表（G0 附带问社区） |
| B.3 | 层表拓扑 | u16 count@0x5A + u64 指针数组（步长 8）@0x78 | **gate 级未知**：可能改内联数组、u32 索引+对象池、std::vector 内联三元组 | scan.rs `try_candidate` 单点改造；NEXT_IDEAS §5.1 已标 Gate 级 | G1：FH6 先验试扫 → 失败转 §C.3 通用拓扑判定 |
| B.4 | Layer 字段 schema | pos f32×2@0x18 / scale@0x28 / rot f32@0x50 / color u8×4@0x74 / mask u8@0x78 / shape_id u8@0x7A；blob ≥0x140 稳定写区 0x80 | 类型序先验 60-70% 保留；**数值按 0-20% 押注**（NEXT_IDEAS §5.1） | layout.layer 全重测 | §C.4 单属性 diff 逐个钉死 |
| B.5 | shape 词表 | 101 矩形 / 102 椭圆 / 228 软椭圆；社区形状 109/114… | 词表需重测：FM 编辑器基础形状集合与 FH6 不保证一致（FM2023 分层编辑器的基础图元集不同） | shape_ids + JSON type 映射重定 | §C.6 词表枚举（编辑器逐形状建层 dump） |
| B.6 | 缩放除数 | 矩形 ÷127，椭圆系 ÷63（=游戏缩放 1.0 对应宽） | 语义可能同构、数值可能不同 | convert.rs 除数来自 layout.scale_divisors（数据） | 模板球已知半径反推（§C.4 步骤 4） |
| B.7 | 坐标系 | pos=(x,−y)；rotation=360°−angle | 同构可能性大，方向/手性待验 | convert.rs 纯函数，改一处带测试 | 非对称排布模板球（球位置我们自己定义）目验 |
| B.8 | 涂装组层上限 | 3000 + 4 边界层 | FM2023 官方上限可能更低（**待社区数字确认，勿引用臆测值**） | 影响大设计拆包策略与模板球球数 | G0 问社区 + 游戏内建到上限实测 |
| B.9 | 反作弊 | 无阻断（fv-inject RPM/WPM 实机可用；三家社区工具存活多年）；ToS 风险声明在案 | **自研反作弊（服务端监控+客户端完整性检查），无 EAC/BattlEye 内核驱动**（2026-10-28 两源一致）；社区存在"EAC"误传与"CE 后台即被检测"的 Horizon 系经验迁移说法 | G0 决定项；写路径风险高于读路径 | ①安装目录查 `EasyAntiCheat/`、`BattlEye/` 文件夹存在性；②运行时进程列表；③trainer 生态现状；④UC 帖反作弊机制描述——四重交叉 |
| B.10 | 写入生效机制 | 即时生效，无 commit（FD6 patterns 注记） | 待验：FM 编辑器 UI 若走脏标记/保存路径，可能需要 commit 步骤（FH 系老版本曾有 POSITION-signature+commit 时代） | write_process.rs 无需改，流程需加"是否即时可见"检查点 | §C.5 写验证首步（写 1 层看渲染） |
| B.11 | 会话局部资源指针 | 0xA8 形状资源指针跨会话必崩，勿写 | 预期同性质字段存在（形状资源引用） | dump 对比跨会话字节（§C.4 对照组） | 同 §C.4；失败模式记录进 STUDY §5 风格台账 |
| B.12 | 平台差异 | FH6 实机为 x64 Steam 类环境 | MS Store/XGP 打包进程（包名进程，OpenProcess 权限路径可能不同）；XGP 版可能禁本地存档干预 | fv-inject 优先支持 Steam 版做 RE；`find_process` 按名列表兼容 | G0 问社区"XGP 版能否 CE"；实机装 Steam 版规避 |

---

## §C 自逆向完整流程（社区无现成表时的 Plan B，同时也是 Plan A 的后备）

原则：**复用 FH6 方法论（STUDY §5）+ 复用 fv-inject 全部安全设施 + 新写 RE 模式工具**。
全程可分四次 5090D 会话完成，每次会话独立产出（dump 语料），崩了不丢进度。

### C.0 阶段零：离线工具补齐（无游戏操作，3-5 天）

| 新写件 | 内容 | 规模 | 复用什么 |
|---|---|---|---|
| `--dump-layers` | 已知布局时按表逐指针 dump 每层 blob（先验 0x140，RE 时放宽 0x400） | ~150 行 | MemIo trait、scan.snapshot_layer |
| `--dump-region` | 布局未知时按区域整块 dump 存档（RE 语料） | ~80 行 | win.rs regions() |
| 结构化 differ | 两次 dump 逐字节 diff → 变化字节簇报告（偏移区间+类型猜测 f32/u16/u8） | ~200 行+测试 | 纯新写，可 Mock 全测 |
| 通用锚点扫描（scan.rs RE 变体） | 不假设 count 偏移：全内存扫 u16==N 的**所有**命中，对每个命中在 ±0x100 窗口内尝试任意偏移读 u64 指针并验证"指向私有可写堆的指针数组"拓扑 | ~250 行 | scan.rs 的区域遍历/ptr 校验骨架 |
| 模板球生成 example | 参数化球数/半径/排布的设计 JSON 生成器（质数球数） | ~80 行 | fv-common Shape/JSON、fv-fit render |
| Mock RE 全链路测试 | 假布局+假内存走完 dump→diff→锁定 全流程 | ~200 行 | MockMem 现成（含 protect 写保护路径） |

### C.1 阶段一：模板球建立 + FH6 先验试扫（会话 1，≤4h）

1. 干净会话（重启游戏、不加载旧设计）→ 涂装编辑器新建涂装组 → 应用 37 层球模板；
2. 先试 **FH6 布局原样扫**（`locate_layer_table` + 现有 `Layout::default()`，仅换 executables）——
   直接命中 = G1 白捡，说明两代偏移未漂（低概率但零成本）；
3. 失败 → 用 C.0 通用锚点扫描（任选 count=37 锚）锁定候选 (group, count_offset, table_offset, 拓扑类型)。

**门**：见 §E G1。

### C.2 阶段二：结构固化 + 字段范围收缩（会话 1 尾或会话 2）

- 锁定后读全表 37 层 blob（`--dump-layers` 0x400 版）存档 = **语料 #1（干净态）**；
- 记录每层指针间距分布（堆分配间距 ≈ blob 实际尺寸线索，FH6 经验 ≥0x140）；
- 表内领域外字段（层间 padding、疑似 vtable/RTTI 指针）记录但不动。

### C.3 阶段三：单属性 diff（会话 2-3，bvzrays 协议照抄，STUDY §5）

每探针 = 改一个属性 → dump → 与上一态 diff。探针预算 15 个（§E G2 判据）：

| 探针序 | 游戏内操作 | 预期特征 |
|---|---|---|
| 0 对照组 ×2 | 什么都不改，间隔 dump 两次 | 排除游戏自身抖动字段（计时器/采样值） |
| 1-3 color | 改某层颜色 R/G/B 各一次 | u8×4 簇中单字节跳变 → 定 color_offset 与通道序（RGB/BGR 由已知颜色定） |
| 4-5 position | 水平/垂直各移动一次 | 两个 f32 各自变化 → 定 pos_offset 与 y 号（已知移动方向定 y 正负 = 坐标系手性） |
| 6-7 scale | 改宽/改高 | f32×2 + 已知目标尺寸反推除数 |
| 8 rotation | 旋转已知角度 | f32 单跳变；已知角度定方向（360−angle 或 +angle） |
| 9 shape_id | 切换基础形状 | u8 突变 → shape_id_offset + 该形状 id 值 |
| 10 alpha | 改透明度（若有 UI） | color 第 4 字节或独立 f32；FH6 经验：基础形状 alpha 恒 255 |
| 11-12 mask/进阶 | 若 UI 有 mask 功能 | mask 字节位置 |
| 13 层增删 | 加一层/删一层 | count ±1 + 表长度变化 → 拓扑复确认 |
| 14 跨会话 | 重启游戏同模板再 dump | 识别会话局部指针字段（FM 版 0xA8 等价物，**列入勿写清单**） |

**纪律**（STUDY §5 原文照抄）：实验写后不存档；每会话结束重启游戏；失败模式逐条记录。

### C.4 阶段四：词表枚举（会话 3，可与 C.3 合并）

编辑器内逐个基础形状建层 → dump shape_id 偏移 → 产出 FM2023 shape_id 表 →
回填 `layout.shape_ids` 与 JSON type 映射；软/渐变形状若存在，alpha 语义单独一探针。

### C.5 阶段五：写验证与固化（会话 4）

1. 同会话小范围写：copy layer0 → layer1（单层、非破坏、可目验）→ 看渲染是否即时生效（B.10 检查点）；
2. 已知设计端到端：37 球模板注入回读 → 游戏内截图比对；
3. 全部通过 → `Layout::save` 产 **fm2023_patterns.json**（仿 fh6_patterns.json 注记 discovered_at/build 号）；
4. 此后 fv-inject 正常三段安全流程（dry-run → 抽样 → `--yes`）整体接管，**无需再改代码**。

### C.6 新写 vs 复用小结

- 复用不动：MemIo/WinMem/MockMem、写入器与回滚、三段安全流程、layout 数据文件机制、
  `Layout::load/save`、convert 换算骨架（数值来自 layout）、`--dump-layout`；
- 改一处（若拓扑变）：scan.rs `try_candidate`；
- 新写：§C.0 六件，合计 ~1000 行 + 测试，3-5 天。

---

## §D 风险矩阵

### D.0 外部事实核查记录（2026-10-28，WebSearch）

- FM2023 反作弊：**非 EAC/BattlEye**，为 Turn 10/微软自研（服务端遥测 + 客户端完整性检查，无内核驱动）；
  Cheat Happens 2023-10 曾称其"比 EAC 略强"导致无 trainer；ForzaMods AIO 存在 "Anti Cheat Failed to Disable"
  issue（证明有客户端侧反作弊模块被 mod 工具尝试绕过）。**但社区存在 EAC 误传**（Reddit 有帖称
  "Game Pass 版用 EAC 绑在线存档"）→ 按 §B.9 四重确认法在 G0 落实，不在本文预断。
- Forza 系反作弊传统（Horizon 系先例）：检测 CE 在后台运行（含未激活状态）可致崩溃/封禁——
  做 RE 时**CE 与 fv-inject 不同时开**，fv-inject 本身不带 CE 特征。
- 封禁主因：经济类（credits/拍卖行）＞ 内存工具本身；社区有"cheating/unallowed modding"永久封禁案例。
- 工具生态：forza-painter 全家族（bvzrays/forza-painter-fh6、eddie772tw/FH6-Painter 等）**仅 Horizon**；
  FM2023 专用内存工具未检索到（G0 待深查 CE 表生态）。
- 来源：[fearlessrevolution.com](https://fearlessrevolution.com)、
  [Nexus Mods — Forza Motorsport (2023)](https://www.nexusmods.com/forzamotorsport/mods/31?tab=posts)、
  [r/forza 封禁/版本讨论](https://www.reddit.com/r/forza/comments/190a0gv/just_got_motorsport_for_pc_but_all_i_want_to_do/)、
  [Steam 讨论区（FH5 反作弊/CE 检测先例）](https://steamcommunity.com/app/1551360/discussions/0/597396334368040282/)、
  [r/cheatengine 表站生态迁移讨论](https://www.reddit.com/r/cheatengine/comments/1p05yf5/what_sites_do_people_use_now_that_fearless)、
  [bvzrays/forza-painter-fh6](https://github.com/bvzrays/forza-painter-fh6)、
  [eddie772tw/FH6-Painter](https://github.com/eddie772tw/FH6-Painter)。

| # | 风险 | 等级 | 依据 | 缓解 |
|---|---|---|---|---|
| R1 | 内存写入触发封号 | **中高** | 自研反作弊 + 客户端完整性检查 + "比 EAC 略强"报告；但 FH6 同类工具多年无系统性封号 | 只读先行（dump 阶段零写入）；小号/非主账号；单人离线场景操作；不碰经济数据；三段安全流程沿用；实验不存档 |
| R2 | CE 后台运行被连带检测 | 中 | FH5 系实证（D.0） | RE 会话中 CE 与注入器绝不同开；fv-inject 无 CE 特征是天然优势 |
| R3 | 客户端完整性检查阻断写入 | 中 | ForzaMods "Anti Cheat Failed to Disable" | G1 前置只读烟测（RPM 全区域可读即通过读闸）；写失败即停不做对抗性绕过（不做 bypass——与 UC 社区路线主动区分，风险可控性优先） |
| R4 | MS Store/XGP 版差异 | 中 | 打包进程权限、二进制版本差 → 布局漂移 | RE 以 Steam 版为准；layout 按 (版本,平台) 命名；XGP 支持列为后续可选 |
| R5 | 层表拓扑改掉 u64 指针数组 | 中 | 六年引擎代差（gate 级未知） | scan.rs 单点改造 ~1-2 天；§C.3 通用拓扑判定兜底 |
| R6 | 层表不在主进程 | 低-中 | 理论可能（FH6 无此问题） | 主进程空手时 procexp 列子进程逐个试扫；`find_process` 已按名列表 |
| R7 | 偏移随游戏更新漂移 | 低 | FD6 先例：结构跨 patch 稳定 | layout JSON 热改；文档记 build 号 |
| R8 | G0 零先例（全自逆） | 中 | forza-painter 家族仅 Horizon | §C 本就按纯自逆设计，成本上限即 3 周；负结果记档不沉没 |
| R9 | 5090D 会话窗口不可得 | 低 | 排期依赖 | 每会话独立产出（语料不丢）；会话次数上限 4 次设计 |
| R10 | ToS/合规 | 低（同 FH6） | 内存修改理论违反 ToS | INJECT.md 风险声明沿用；只开源工具与文档，不分发任何游戏内产物；不做对抗反作弊 |

---

## §E go/no-go 门与判据

### G0 社区可行性（48h 预算，零成本，可立即执行）

| 判定 | 判据 |
|---|---|
| **GO（带表）** | 找到 ≥1 张通过 §A.4 强特征核验的含涂装层表资源 → 直接到 G1（表作为拓扑答案+偏移先验，**不作为免验通行证**——偏移随版本漂移，G1 必走） |
| **WEAK-GO（无表，转纯自逆）** | 无层表资源，但反作弊旁证 ≥2 项：trainer 存在 / CE 附加无即时封禁先例 / 只读 dump 先例 / mod 工具日常读写内存 → 按 §C 全流程走 |
| **NO-GO** | 48h 内 Q1 零命中 **且** Q2 证据链为负（CE 附加即踢/即封、无任何内存工具先例、客户端完整性阻断读）→ 整项冻结记负结果，资源转 FH7 day-one 工具链（该产出对 FH7 同样有效，无沉没成本） |

### G1 模板球一击（会话 1 内，≤4h 墙钟）

| 判定 | 判据 |
|---|---|
| **GO** | 只读烟测通过（OpenProcess + RPM 区域可读）且 FH6 布局先验直接命中，或通用锚点扫描在会话内锁定 (count_offset, table 拓扑) |
| **PARTIAL** | count 锚点已锁但拓扑非 u64 指针数组 → 评估 scan.rs 改造（预估 1-2 天）后继续；表结构语料归档 |
| **NO-GO** | 读路径不通（反作弊阻断读）或 3 次会话仍无锚点 → 停止，发布 RE 工具 + dump 语料（社区贡献），注入支持无限期延后 |

### G2 单属性定位（会话 2-3，≤15 个 diff 探针）

| 判定 | 判据 |
|---|---|
| **GO** | position/color/scale/rotation/shape_id 五字段 ≥4 个钉死且写验证渲染即时生效、游戏不崩 → 固化 fm2023_patterns.json，端到端验证 |
| **PARTIAL** | 钉死 <4 字段或写路径不稳 → 冻结写能力，发布"layout-unknown 工具 + dump 语料 + 已知偏移"（仍是对社区的真实贡献） |
| **NO-GO** | 写入即崩且失败模式不可控（对照 STUDY §5 台账无法隔离）→ 只读模式封版 |

---

## §F 时间线（part-time，1.5-3 周 + 前置 G0）

```
W0（本周，0 成本）   G0 调研 48h（§A 协议）→ G0 判据打分 → 立项/冻结决定
W1（3-5 天）         §C.0 离线工具补齐（六件 ~1000 行，Mock 全测，不动游戏）
W2（会话 1-2）       会话 1：模板球 + G1 + 干净态 dump；会话 2：diff 探针批 1
                     （对照组/color/pos/scale，~8 探针）
W3（会话 3-4）       会话 3：rotation/shape_id/词表枚举/跨会话对照（~7 探针）；
                     会话 4：写验证 + fm2023_patterns.json 固化 + 端到端注入
缓冲 +50%            G0 为 WEAK-GO（纯自逆）或拓扑 PARTIAL 时取上限 3 周
```

- 排期约束（沿用 NEXT_IDEAS §6）：主体排在开源发布之后；5090D 窗口空闲期；
  GPU 任务与本项无冲突（RE 全程 CPU/内存 I/O）；
- 每会话退出纪律：不存档、重启游戏、失败模式记档（STUDY §5 台账格式）；
- 产出物清单：G0_FINDINGS.md → RE 模式六件套（入 fv-inject）→ dump 语料库 →
  fm2023_patterns.json → INJECT.md FM 附录。
- 战略对冲（NEXT_IDEAS §5.3 原文）：无论 FM 成败，"layout-unknown RE 模式 + 数据驱动壳"
  都是 **FH7 day-one 就绪能力**——本项目的沉没成本下限是工具链本身。

---

*本文档为规划产物；所有"预期"列均为待验证假说，不构成事实主张。首次实机操作前须重新核对游戏版本与 §D.0 外部事实。*
