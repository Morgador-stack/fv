#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""远端对账 + 补传：对比本地 HEAD 树与 GitHub 远端树（blob SHA 级精确对比），
缺失/不一致的文件经 Contents API 逐个 PUT 补齐（每文件一个 commit）。
用法：python tools/sync_finish.py [--dry-run]
绕开点：github.com:443 push 被墙，api.github.com 可达；
        细粒度 PAT 无 tree/commit/ref 写权限，故不走 Git Data 写端点。
排除：.github/workflows/*（需 workflow scope，PAT 未授予）。
"""
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

API_HOST = "api.github.com"
API = f"https://{API_HOST}/"
OWNER, REPO = "Morgador-stack", "fv"
BRANCH = "main"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
assert SAFE_NAME.match(OWNER) and SAFE_NAME.match(REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIT = r"D:\Tools\MinGit\cmd\git.exe"
SKIP_PREFIX = (".github/workflows/",)


def call(method, path, token, body=None):
    assert path.startswith("/"), f"非法路径: {path}"
    url = API + path[1:]
    assert url.startswith(f"https://{API_HOST}/"), f"越界 URL: {url}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "fv-sync",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    dry = "--dry-run" in sys.argv
    token = (
        open(os.path.join(ROOT, ".token")).read().strip()
        if os.path.exists(os.path.join(ROOT, ".token"))
        else os.environ.get("GH_TOKEN", "")
    )
    assert token, "无 token：.token 文件 / GH_TOKEN"

    # 本地树：path -> (mode, sha)
    out = subprocess.run(
        [GIT, "ls-tree", "-r", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    local = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, _typ, sha = meta.split()
        local[path] = (mode, sha)
    print(f"[local] HEAD 树 {len(local)} 文件")

    # 远端树
    st, c = call("GET", f"/repos/{OWNER}/{REPO}/commits/{BRANCH}", token)
    assert st == 200, f"HEAD {st}: {c}"
    print(f"[remote] HEAD {c['sha'][:10]} {c['commit']['message'].splitlines()[0]}")
    st, t = call(
        "GET",
        f"/repos/{OWNER}/{REPO}/git/trees/{c['commit']['tree']['sha']}?recursive=1",
        token,
    )
    assert st == 200, f"tree {st}: {t}"
    assert not t.get("truncated"), "远端树被截断，需分页"
    remote = {
        e["path"]: (e["mode"], e["sha"])
        for e in t["tree"] if e["type"] == "blob"
    }
    print(f"[remote] 树 {len(remote)} 文件")

    missing = sorted(p for p in local if p not in remote)
    changed = sorted(
        p for p in local
        if p in remote and remote[p][1] != local[p][1]
    )
    extra = sorted(p for p in remote if p not in local)
    wf_missing = [p for p in missing if p.startswith(SKIP_PREFIX)]
    todo = [p for p in missing + changed if not p.startswith(SKIP_PREFIX)]
    print(f"[diff] 缺失 {len(missing)}  内容不同 {len(changed)}  远端多余 {len(extra)}")
    if wf_missing:
        print(f"[skip] workflow 文件 {len(wf_missing)}（PAT 无 workflow scope）: {wf_missing}")
    for p in extra:
        print(f"[extra] 远端多出: {p}")

    if dry:
        for p in todo:
            print(f"  [todo] {p}")
        print(f"[dry-run] 需补传 {len(todo)} 文件，未执行")
        return

    ok, fail = 0, 0
    for i, path in enumerate(todo, 1):
        # 直读 HEAD blob（字节级精确）：工作区经 autocrlf 换行转换，裸读会失配
        raw = subprocess.run(
            [GIT, "cat-file", "blob", local[path][1]], cwd=ROOT,
            capture_output=True, check=True,
        ).stdout
        body = {
            "message": f"sync: {path}",
            "content": base64.b64encode(raw).decode(),
            "branch": BRANCH,
        }
        if path in remote:
            body["sha"] = remote[path][1]
        st, r = call("PUT", f"/repos/{OWNER}/{REPO}/contents/{path}", token, body)
        if st in (200, 201):
            ok += 1
            print(f"  [{i}/{len(todo)}] {path} ✓")
        else:
            fail += 1
            print(f"  [{i}/{len(todo)}] {path} ✗ {st}: {json.dumps(r)[:200]}")

    print(f"[done] 成功 {ok}  失败 {fail}")
    if ok and not fail:
        print(f"[repo] https://github.com/{OWNER}/{REPO}")


if __name__ == "__main__":
    main()
