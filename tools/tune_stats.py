# tune v2/v3 log stats: per-knob tries/wins/gains (TUNE_V2_ANALYSIS data base)
# + v3 additions: vertex-eval adoption rate (TUNE_V3_DESIGN D.4-3) and
#   per-image eval counts (D.4-2). Usage:
#   python tools/tune_stats.py [glob]   # default: bench/mve0/t_*.log
import re, glob, json, os, sys

DEFAULT_GLOB = r"D:/WorkSpace/ZCWorkspace/forza-vinyl/bench/mve0/t_*.log"
pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB

knob = {}
vtx = {"tries": 0, "wins": 0}  # v3 vertex-eval adoption (lines tagged " (vertex)")
imgs = []
evals = {}  # per-image eval count = knob lines + 1 baseline (inf-psnr lines are
            # not regex-matchable, so images that hit inf mid-search may undercount)
knob_re = re.compile(
    r"\[tune\] (\w+)\s+=(-?[\d.]+)\s+psnr=([\d.]+) shapes=(\d+)(\s+.. best)?"
)
for log in sorted(glob.glob(pattern)):
    name = os.path.basename(log)
    name = name[2:-4] if name.startswith("t_") else os.path.splitext(name)[0]
    txt = open(log, encoding="utf-8", errors="ignore").read()
    imgs.append(name)
    cur_best = -1e9
    n_evals = 0
    for line in txt.splitlines():
        m = knob_re.search(line)
        if not m:
            continue
        knob_name = m.group(1)
        psnr = float(m.group(3))
        win = bool(m.group(5))
        vertex = "(vertex)" in line  # v3 顶点验证评估标记（§D.2 日志兼容）
        n_evals += 1
        if vertex:
            vtx["tries"] += 1
        d = knob.setdefault(knob_name, {"tries": 0, "wins": 0, "gains": []})
        d["tries"] += 1
        if win:
            d["wins"] += 1
            d["gains"].append(psnr - cur_best)
            if vertex:
                vtx["wins"] += 1
        if win:
            cur_best = psnr
    evals[name] = n_evals + 1  # + baseline evaluation

print("images:", len(imgs))
print(f"{'knob':<16}{'tries':>6}{'wins':>6}{'winrate':>9}{'avg_gain':>10}")
for k, d in knob.items():
    wr = 100 * d["wins"] / max(1, d["tries"])
    ag = sum(d["gains"]) / len(d["gains"]) if d["gains"] else 0.0
    print(f"{k:<16}{d['tries']:>6}{d['wins']:>6}{wr:>8.1f}%{ag:>9.3f}dB")

vrate = 100 * vtx["wins"] / max(1, vtx["tries"])
print(f"vertex: {vtx['wins']}/{vtx['tries']} adopted ({vrate:.1f}%)  [D.4-3 target >5%]")
if evals:
    vals = sorted(evals.values())
    mean = sum(vals) / len(vals)
    print(f"evals/image: mean={mean:.1f} min={vals[0]} max={vals[-1]}  [D.4-2 target mean<=34, max<=39]")

rounds = {}
for log in sorted(glob.glob(pattern)):
    txt = open(log, encoding="utf-8", errors="ignore").read()
    parts = re.split(r"\[tune\] round (\d+)", txt)
    for i in range(1, len(parts), 2):
        rd = int(parts[i])
        wins = len(re.findall(r"best", parts[i + 1]))
        rounds[rd] = rounds.get(rd, 0) + wins
print("round wins:", dict(sorted(rounds.items())))

# JSON 产物随语料目录走：默认 = bench/mve0/knob_stats.json（v2 行为不变）；
# 传入自定义 glob（如 v3 新日志目录）时写到该目录，不覆盖主语料产物。
out_json = os.path.join(os.path.dirname(os.path.abspath(pattern.split("*")[0])), "knob_stats.json")
with open(out_json, "w") as f:
    json.dump({"knobs": knob, "round_wins": rounds, "vertex": vtx, "evals": evals}, f, indent=1)
print("saved", out_json)
