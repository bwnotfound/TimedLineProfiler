# Timed Line Profiler

带**时间窗口**的逐行性能分析器，专为 ML 训练/推理脚本设计。

与传统 `line_profiler`（kernprof）的区别：

| 特性                      | line_profiler     | timed-line-profiler  |
| ------------------------- | ----------------- | -------------------- |
| 行级耗时                  | ✅                 | ✅                    |
| 不修改训练脚本            | ❌ 需要 `@profile` | ✅ 通过 runner 启动   |
| 时间段维度（漂移分析）    | ❌ 只有总和        | ✅ bucket 聚合        |
| GPU 异步处理              | ❌                 | ✅ 可选 `--cuda-sync` |
| 起止触发器（跳过 warmup） | ❌                 | ✅ `FILE:LINE:N`      |
| 多文件 glob               | ❌                 | ✅                    |
| HTML 可视化               | ❌                 | ✅ plotly             |

## 安装

### 方式一：从 git 安装（推荐内部使用）

```bash
pip install git+ssh://git@your.git.host/yourname/LineProfiler.git
# 或 https
pip install git+https://your.git.host/yourname/LineProfiler.git
```

升级直接重跑同一条命令；锁定版本可加 `@v0.1.0` 或 `@<commit>`。

### 方式二：本地开发安装

```bash
git clone <repo> && cd LineProfiler
pip install -e .             # editable，改代码即生效
pip install -e ".[torch]"    # 同时拉 torch（可选）
```

### 方式三：clone 后直接跑（不安装）

```bash
git clone <repo> && cd LineProfiler
python -m timed_line_profiler --target xxx.py --out report.html -- xxx.py
```

只要在仓库根目录下执行，Python 会把当前目录加进 `sys.path`，
不需要 `pip install` 即可运行。**前提是 `plotly` 已经在环境里**（HTML 输出依赖）。

## 使用

安装后多了一个 `tlprof` 命令：

```bash
# 最小用法
tlprof --target your_model.py --out report.html -- train.py

# 完整用法（GPU 训练，跳过 5 次 warmup，profile 100 步，glob 多文件）
tlprof \
    --target 'model/**/*.py' --target 'trainer/loop.py' \
    --bucket 5.0 --top-k 30 \
    --start-at trainer/loop.py:120:5 \
    --stop-at  trainer/loop.py:120:105 \
    --cuda-sync \
    --out report.html --txt report.txt \
    -- trainer/loop.py --epochs 100
```

`--` 之后的所有参数会原样传给训练脚本。

### 作为库使用

```python
from timed_line_profiler import TimedLineProfiler, render_html, render_text

profiler = TimedLineProfiler(
    target_files={"/abs/path/to/model.py"},
    bucket_seconds=5.0,
    cuda_sync=True,
)
profiler.start()
try:
    train(...)
finally:
    profiler.stop()

print(render_text(profiler))
render_html(profiler, "report.html", top_k=30)
```

## 关键参数

| 参数                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `--target`               | 目标文件，支持 glob（`model/**/*.py`），可重复               |
| `--bucket`               | 时间窗口大小（秒），影响热力图横轴粒度                       |
| `--start-at FILE:LINE:N` | 该行命中第 N 次时开始记录（跳过 import/warmup）              |
| `--stop-at FILE:LINE:M`  | 该行命中第 M 次时停止记录                                    |
| `--cuda-sync`            | 每个目标行后调用 `torch.cuda.synchronize()`，测真实 GPU 耗时 |
| `--out`                  | HTML 输出路径                                                |
| `--txt`                  | 文本报告输出路径（可选）                                     |
| `--top-k`                | 可视化 Top-K 行                                              |
| `--threshold-ms`         | 文本报告中过滤总耗时低于该值的行                             |

## 已知局限

- 子进程（如 DataLoader workers）不会被追踪，只追踪主进程
- `sys.settrace` 自带 5–30x slowdown；`--cuda-sync` 会进一步串行化 CPU/GPU
- 与 `pdb` / `debugpy` 等同样使用 `sys.settrace` 的工具不能并存
- DDP 多 rank 训练时，每个 rank 都会写 HTML，请加 RANK 区分输出路径

## 开发

```bash
pip install -e ".[dev]"
python -m build         # 打 wheel/sdist
twine upload dist/*     # 发布（如有 PyPI 账号）
```

## License

MIT，见 [LICENSE](./LICENSE)。
