#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""经 api.github.com 的 Git Data API 全量上传仓库（绕开被墙的 github.com:443 push 通道）。
用法：python gh_upload.py <owner> <repo> [token]
  token 缺省读 <repo根>/.token 文件（或环境变量 GH_TOKEN）。
流程：建仓（若缺）→ git ls-tree 取全部文件 → 逐文件 POST blob →
      POST tree → POST commit → PATCH refs/heads/main。
安全：URL 只允许 https://api.github.com/ 前缀（常量），owner/repo 仅
      [A-Za-z0-9_.-]，禁自动重定向（防 SSRF/DNS rebinding）。
"""
import base64
import json
import os
import re
import subprocess
import sys
import urllib.request

API_HOST = "api.github.com"
API = f"https://{API_HOST}/"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect blocked", headers, fp)


def call(method, path, token, body=None):
    # SSRF 守卫：path 只能是 / 开头的相对路径，最终 URL 钉死在 api.github.com
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
            "User-Agent": "fv-upload",
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    owner, repo = sys.argv[1], sys.argv[2]
    assert SAFE_NAME.match(owner) and SAFE_NAME.match(repo), "owner/repo 仅允许 [A-Za-z0-9_.-]"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    token = (
        sys.argv[3]
        if len(sys.argv) > 3
        else open(os.path.join(root, ".token")).read().strip()
        if os.path.exists(os.path.join(root, ".token"))
        else os.environ.get("GH_TOKEN", "")
    )
    assert token, "无 token：参数3 / .token 文件 / GH_TOKEN 环境变量三选一"

    st, me = call("GET", "/user", token)
    assert st == 200, f"/user {st}: {me}"
    assert me["login"] == owner, f"token 属于 {me['login']}，与参数 owner={owner} 不符"
    print(f"[auth] {me['login']} ✓")

    st, r = call("POST", "/user/repos", token, {
        "name": repo, "private": False,
        "description": "图片 → Forza vinyl 图层（B1 结构采样 · 可微渲染 · polish 循环 · 40.24dB）",
        "auto_init": False,
    })
    if st == 201:
        print(f"[repo] {owner}/{repo} 已创建")
    elif st == 422:
        print(f"[repo] {owner}/{repo} 已存在，复用")
    else:
        sys.exit(f"[repo] 失败 {st}: {r}")

    out = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout
    entries = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split()
        assert typ == "blob"
        # workflow 文件需要 token 的 workflow 权限（首枚 PAT 只有 repo）——
        # 排除后经浏览器会话直传（上传文件无需 token）。
        if path.startswith(".github/workflows/"):
            continue
        entries.append((mode, sha, path))
    print(f"[tree] {len(entries)} 个文件（workflow 文件排除）")

    tree_items = []
    primed = False
    for i, (mode, sha, path) in enumerate(entries):
        raw = subprocess.run(
            ["git", "cat-file", "blob", sha], cwd=root,
            capture_output=True, check=True,
        ).stdout
        st, b = call("POST", f"/repos/{owner}/{repo}/git/blobs", token, {
            "content": base64.b64encode(raw).decode(), "encoding": "base64",
        })
        if st == 409 and not primed:
            # 空仓陷阱：Git Data API 拒绝零提交仓库；Contents API 又建不了嵌套
            # 路径。用树里第一个根级文件（无 '/'）造初始提交，再回到 blob 流程。
            prime_mode, prime_sha, prime_path = next(
                (m, s, p) for m, s, p in entries if "/" not in p
            )
            raw_prime = subprocess.run(
                ["git", "cat-file", "blob", prime_sha], cwd=root,
                capture_output=True, check=True,
            ).stdout
            st2, b2 = call("PUT", f"/repos/{owner}/{repo}/contents/{prime_path}", token, {
                "message": "bootstrap (empty-repo prime)",
                "content": base64.b64encode(raw_prime).decode(),
                "branch": "main",
            })
            assert st2 in (200, 201), f"prime {prime_path}: {st2} {b2}"
            primed = True
            st, b = call("POST", f"/repos/{owner}/{repo}/git/blobs", token, {
                "content": base64.b64encode(raw).decode(), "encoding": "base64",
            })
        assert st == 201, f"blob {path}: {st} {b}"
        assert b["sha"] == sha, f"SHA 不一致 {path}"
        tree_items.append({"path": path, "mode": mode, "type": "blob", "sha": sha})
        if (i + 1) % 25 == 0:
            print(f"  blobs {i+1}/{len(entries)}")
    print(f"[blobs] {len(tree_items)} ✓（SHA 全一致）")

    st, t = call("POST", f"/repos/{owner}/{repo}/git/trees", token, {"tree": tree_items})
    assert st == 201, f"tree {st}: {t}"
    st, c = call("POST", f"/repos/{owner}/{repo}/git/commits", token, {
        "message": "fv: full import via Git Data API\n\n"
                   "12-commit local history preserved in fv.bundle; see STUDY.md.",
        "tree": t["sha"], "parents": [],
    })
    assert st == 201, f"commit {st}: {c}"
    st, ref = call("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/main", token, {
        "sha": c["sha"], "force": True,
    })
    if st == 422:
        st, ref = call("POST", f"/repos/{owner}/{repo}/git/refs", token, {
            "ref": "refs/heads/main", "sha": c["sha"],
        })
    assert st in (200, 201), f"ref {st}: {ref}"
    print(f"[done] https://github.com/{owner}/{repo} ← commit {c['sha'][:8]}（{len(tree_items)} 文件）")


if __name__ == "__main__":
    main()
