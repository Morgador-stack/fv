# 推送到 GitHub（给用户的操作卡）

仓库已就绪：`forza-vinyl/` 内 git 仓库，2 个 commit（源码+文档+基准），打包 359KB。
网络现状（2026-10-27 21:00 实测）：**GitHub 直连可达**（github.com / api.github.com 均 200）。
缺的只是你的账号凭据。三条路径任选：

## 路径 A：PAT 一行推送（最快，推荐）

1. 浏览器开 https://github.com/new → 建空仓（名字建议 `fv` 或 `forza-vinyl`，Private/Public 任选，**不要**勾选 README/license/gitignore——保持空仓）
2. 生成 token：https://github.com/settings/tokens?type=beta → Generate → 勾选该仓库的 Contents: Read and write
3. 在本目录（D:\WorkSpace\ZCWorkspace\forza-vinyl）执行：

```bash
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
# 弹出认证时：用户名=<用户名>，密码=<粘贴 PAT>
```

## 路径 B：SSH（若你有现成密钥对）

```bash
git remote add origin git@github.com:<用户名>/<仓库名>.git
git push -u origin main
```

## 路径 C：离线搬运（fv.bundle，359KB）

U 盘/网盘带走 `fv.bundle`，在任意有网机器：

```bash
git clone fv.bundle fv && cd fv && git remote set-url origin https://github.com/<u>/<repo>.git && git push -u origin main
```

## 推完之后（可选）

- `gh repo set-default` + 装 gh CLI 后，Release 附件挂 `bundle-5090d/` 打包的 zip（含二进制/模型/设计 JSON，254MB）
- CI 已配（.github/workflows/ci.yml：windows x64 跑 fv-common+fv-fit 测试+clippy）
- 注意：仓库已排除 refs/（第三方）、deps/（模型/dll）、全部 target；STUDY.md 内含开发机路径与逆向偏移讨论，公开发布前可再过目一遍
