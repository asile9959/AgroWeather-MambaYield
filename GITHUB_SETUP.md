# GitHub 接入说明

当前项目已经补充了 `README.md`、`.gitignore` 和 `requirements.txt`。

## 推荐方式：在 WSL2/Linux 里上传

进入项目目录：

```bash
cd /mnt/c/Users/sc/Documents/Codex/2026-05-17/files-mentioned-by-the-user-cm
```

初始化 Git：

```bash
git init
git branch -M main
git add README.md .gitignore requirements.txt GITHUB_SETUP.md \
  weather_yield_experiment.py \
  weather_yield_deep_experiment.py \
  run_weather_yield_ablations.py \
  *.md
git commit -m "Initial weather-yield forecasting project"
```

连接远程仓库：

```bash
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

## 注意

`.gitignore` 默认忽略了实验输出、模型权重和数据文件，避免把大文件或原始数据传到 GitHub。

如果确实要上传某个小型示例数据集，可以放到 `examples/` 目录，并在 `.gitignore` 里单独放行。
