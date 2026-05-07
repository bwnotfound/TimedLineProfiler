# Timed Line Profiler

带**时间窗口**的逐行性能分析器，专为 ML 训练/推理脚本设计。

| 特性                                               | line_profiler | timed-line-profiler |
| -------------------------------------------------- | ------------- | ------------------- |
| 行级耗时                                           | ✅             | ✅                   |
| 不修改训练脚本                                     | ❌             | ✅                   |
| 时间段维度（漂移分析）                             | ❌             | ✅ bucket 聚合       |
| GPU 异步处理                                       | ❌             | ✅ `--cuda-sync`     |
| 起止触发器                                         | ❌             | ✅                   |
| 多文件 glob                                        | ❌             | ✅                   |
| **第三方库相对路径模式**                           | ❌             | ✅                   |
| **gitignore 风格 exclude**                         | ❌             | ✅                   |
| **每文件 top-k + ratio 并集选行**                  | ❌             | ✅                   |
| HTML 可视化（含 log scale 自适应 + 文件 dropdown） | ❌             | ✅                   |
| Markdown 完整聚合报告                              | ❌             | ✅                   |

## 安装

```bash
# 方式一：从 git 安装
pip install git+ssh://git@your.git.host/yourname/LineProfiler.git

# 方式二：本地开发
git clone <repo> && cd LineProfiler
pip install -e .

# 方式三：clone 后直接跑（不安装，需要 plotly 已在环境里）
git clone <repo> && cd LineProfiler
python -m timed_line_profiler --target xxx.py --out ./report_dir -- xxx.py
```

## 快速使用

```bash
# 最小用法（--out 是目录，不存在时自动创建）
tlprof --target your_model.py --out ./report_dir -- train.py
```

输出目录里会有：
- `report.html` — plotly 交互式报告（热力图 + 折线图 + 汇总表，自适应 log scale）
- `report.md` — 相对完整的聚合数据（按文件全行展开 + bucket 矩阵）

## `--target` 的两种语义

每个 `--target` 模式按以下顺序解析：

1. **FS glob 优先**：在当前 cwd 下用 `glob.glob(pat, recursive=True)` 查找具体 `.py` 文件，命中即按绝对路径加入目标集合。
2. **相对路径 glob 兜底**：若 FS 下完全没命中，则编译成 regex，runtime 中按"末尾路径段"匹配。这种语义不依赖 cwd 是否存在该文件。

第 (2) 种用于 profile **已安装的第三方库**——只需提供库里文件**相对库根目录**的路径：

```bash
# Profile pandas 里某个文件
tlprof --target 'pandas/core/frame.py' --out ./rep -- my_script.py

# Profile pandas/core 下所有 .py
tlprof --target 'pandas/core/*.py' --out ./rep -- my_script.py

# Profile torch.nn.modules 整个模块
tlprof --target 'torch/nn/modules/**/*.py' --out ./rep -- train.py

# 多个 target 并存
tlprof \
    --target 'mymodel/**/*.py' \
    --target 'torch/nn/modules/linear.py' \
    --out ./rep -- train.py
```

支持的 glob 语法：`*`（不跨 `/`）、`**`（跨多层）、`?`、`[...]`、`[!...]`。

`--target` / `--exclude` / `--exclude-from` 都支持**逗号分隔**多个值，与重复 flag 等价：

```bash
# 这两条命令完全等价
tlprof --target 'mymodel/**/*.py,torch/nn/modules/linear.py' --out ./rep -- train.py
tlprof --target 'mymodel/**/*.py' --target 'torch/nn/modules/linear.py' --out ./rep -- train.py
```

> ⚠️ 不支持文件名含 `,`（极罕见）；前后空格会被自动 strip，空项被忽略。

> ⚠️ "FS 命中优先" 意味着：cwd 下 FS 已能匹配的模式不会再 fallback 到相对模式。
> 这避免了 `**/*.py` 这类宽泛模式作为相对模式时误伤整个文件系统的所有 `.py`。

## `--exclude` / `--exclude-from`

- `--exclude PATTERN`：排除文件，与 `--target` 同 glob 语法；可重复，单个内可用逗号分隔
- `--exclude-from FILE`：从 gitignore 风格文件读取（每行一个模式）；可重复，单个内可用逗号分隔多个文件路径
- exclude 优先级 > target，命中即不追踪

```bash
# 直接排除模式
tlprof --target 'src/**/*.py' --exclude 'src/tests/**' \
    --out ./rep -- main.py

# 从 .gitignore 读取
tlprof --target 'src/**/*.py' --exclude-from .gitignore \
    --out ./rep -- main.py

# 两者并用
tlprof --target 'src/**/*.py' \
    --exclude '**/test_*.py' \
    --exclude-from .gitignore \
    --exclude-from .profignore \
    --out ./rep -- main.py
```

支持的 gitignore 语法：

| 语法      | 行为                                             |
| --------- | ------------------------------------------------ |
| `# foo`   | 注释，跳过                                       |
| 空行      | 跳过                                             |
| `foo/`    | 目录，等价于 `foo/**`（排除该目录下所有）        |
| `/foo`    | 去掉前导 `/` 当作 `foo` 处理（按目录段开头匹配） |
| `foo`     | 在任意目录段开头匹配                             |
| `**/*.py` | 标准 glob                                        |
| `!foo`    | **不支持**否定语法，会跳过该规则并 warn          |

## `--start-at` / `--stop-at` / `--profile-hits` / `--max-duration`

四个参数共同控制"什么时候开始/停止记录"。

**开始**：

- `--start-at FILE:LINE[:N]`：当 FILE 第 LINE 行被命中第 N 次时开始记录；N 缺省为 1（首次命中即开始）。FILE 同样支持 glob/相对路径。
- 不指定 `--start-at` → 程序一启动就记录。

**停止**（以下三者**任一先达到**就停止，OR 关系，与口语"取 min"等价）：

- `--stop-at FILE:LINE[:M]`：FILE 第 LINE 行被命中第 M 次时停止；M 缺省 1（绝对计数，从程序启动起算）。
- `--profile-hits M`：开始记录后，**`--start-at` 那行**被再命中 M 次时停止（相对 start 计数）。需配合 `--start-at` 使用。
- `--max-duration SEC`：开始记录后最多 SEC 秒就停止。

举例："profile 主循环 100 步" 是常见需求：

```bash
tlprof --target 'mymodel/**/*.py' \
    --start-at 'mymodel/trainer/loop.py:120' \
    --profile-hits 100 \
    --out ./rep -- train.py
```

举例："warmup 5 步后 profile 100 步，但最多 30 秒"：

```bash
tlprof --target 'mymodel/**/*.py' \
    --start-at 'mymodel/trainer/loop.py:120:5' \
    --profile-hits 100 --max-duration 30 \
    --out ./rep -- train.py
```

> ⚠️ 当 `--start-at` / `--stop-at` / `--profile-hits` 涉及的 file 模式命中**多个文件**时，"第 N 次"是**全局累加**（跨文件计数）。要精确触发，请用具体单文件路径。

> ⚠️ stop 之后 `--start-at` **不会再次触发**（防止 stop → start → stop 反复跳转）。如果你的 trigger 行就是循环体内某行，这一点尤其重要。

## 选行规则：每文件 `max(top-k, ratio 触发数)`

可视化（HTML 热力图/折线/表格 + Markdown bucket 矩阵）展示的行按下面规则选：

- 每个文件按耗时降序
- 选前 `max(K, M)` 行，其中：
  - `K = --top-k`：每文件保底选 K 行
  - `M = --top-ratio` 触发的行数：占该文件总耗时 ≥ ratio% 的行数（因占比单调下降，等价于"前 M 行"）
- 全部选中行的全局并集 = 最终展示集

含义：
- top-k 是下限，保证小文件也有代表行被选中（不会被大文件淹没）
- ratio 让占比高的大文件按需扩展更多行

```bash
# 每文件至少 10 行 + 占比 ≥ 1% 的也选上
tlprof --target 'src/**/*.py' \
    --top-k 10 --top-ratio 1.0 \
    --out ./rep -- main.py
```

`--top-ratio` 默认 0（仅 top-k 生效）。

## `--out` 输出目录

- 不存在时**自动创建**；若已存在但不是目录 → 立刻 `exit 2`
- 训练前完成创建/校验，避免训练完才发现写不进去
- 内含：
  - `report.html` — plotly 交互式报告
    - **顶部文件 dropdown**：默认"全部文件"视图，可切换到任一单文件视图，热力图/折线/表格/标题同步更新
    - **dropdown 限制 top-N 文件**（默认 8，可用 `--html-max-file-views` 调整，或 `--html-all-files` 全量）：避免文件多时 HTML 体积爆炸；完整数据始终在 `report.md` 里
    - 热力图、折线图在动态范围 > 50 倍时**自动启用 log scale**（避免 outlier 压扁画面）
    - 高度按选中行数自适应
  - `report.md` — 完整聚合数据
    - 概览（总耗时/命中数/文件数/窗口数）
    - 各文件全行表（按行号升序）
    - 选中行的 bucket 分布矩阵

## 完整示例

```bash
tlprof \
    --target 'model/**/*.py' \
    --target 'trainer/loop.py' \
    --target 'torch/nn/modules/*.py' \
    --exclude '**/test_*.py' \
    --exclude-from .gitignore \
    --bucket 5.0 \
    --top-k 10 --top-ratio 1.0 \
    --start-at trainer/loop.py:120:5 \
    --profile-hits 100 --max-duration 60 \
    --cuda-sync \
    --out ./report_dir \
    -- trainer/loop.py --epochs 100
```

`--` 之后的所有参数会原样传给训练脚本。

## 作为库使用

```python
from timed_line_profiler import (
    TimedLineProfiler, render_html, render_markdown, render_text,
)
from timed_line_profiler.pattern import glob_to_regex, parse_exclude_file

profiler = TimedLineProfiler(
    target_files={"/abs/path/to/model.py"},
    target_patterns=[
        ("torch/nn/modules/*.py", glob_to_regex("torch/nn/modules/*.py")),
    ],
    exclude_patterns=[
        ("**/test_*.py", glob_to_regex("**/test_*.py")),
        *parse_exclude_file(".gitignore"),
    ],
    bucket_seconds=5.0,
    cuda_sync=True,
)
profiler.start()
try:
    train(...)
finally:
    profiler.stop()

print(render_text(profiler))
render_html(profiler, "rep/report.html", top_k=10, top_ratio_pct=1.0)
render_markdown(profiler, "rep/report.md", top_k=10, top_ratio_pct=1.0)
```

## 模块结构

```
timed_line_profiler/
├── __init__.py        # 公共 API 导出
├── __main__.py        # python -m 入口
├── pattern.py         # 路径模式工具（glob → regex、target/exclude/trigger 解析）
├── profiler.py        # 核心 TimedLineProfiler 类
├── selection.py       # 行选择策略（select_keys_per_file）
├── cli.py             # 命令行入口
└── reports/           # 三种格式的报告渲染
    ├── __init__.py    # re-export render_text / render_markdown / render_html
    ├── text.py
    ├── md.py
    └── html.py        # 含文件 dropdown 切换
```

## 关键参数速查

| 参数                       | 说明                                                                  |
| -------------------------- | --------------------------------------------------------------------- |
| `--target`                 | 目标文件；可重复，单个内可用逗号分隔；FS glob 优先 + 相对路径模式兜底 |
| `--exclude`                | 排除文件 glob；可重复，单个内可用逗号分隔                             |
| `--exclude-from`           | gitignore 风格排除规则文件；可重复，单个内可用逗号分隔                |
| `--bucket`                 | 时间窗口大小（秒）                                                    |
| `--start-at FILE:LINE[:N]` | 该行命中第 N 次时开始记录（N 默认 1）                                 |
| `--stop-at FILE:LINE[:M]`  | 该行命中第 M 次时停止记录（M 默认 1，绝对计数）                       |
| `--profile-hits M`         | 开始记录后 --start-at 那行再命中 M 次后停止（相对计数）               |
| `--max-duration SEC`       | 开始记录后最多 SEC 秒就停止；与上述 stop 条件 OR 关系                 |
| `--cuda-sync`              | 每个目标行后调 `torch.cuda.synchronize()`                             |
| `--out`                    | 输出目录（不存在时自动创建），生成 report.html + report.md            |
| `--top-k`                  | 每个文件至少选 k 行（默认 10）                                        |
| `--top-ratio PCT`          | 占文件总耗时 ≥ PCT% 的行也选上，与 top-k 取并集（默认 0）             |
| `--threshold-ms`           | 报告中过滤总耗时低于该值的行                                          |
| `--html-max-file-views N`  | HTML dropdown 中最多列出 top-N 个单文件视图（默认 8）                 |
| `--html-all-files`         | HTML dropdown 列出全部文件（覆盖 max-file-views，HTML 会变大）        |

## 已知局限

- 子进程（如 DataLoader workers）不会被追踪，只追踪主进程
- `sys.settrace` 自带 5–30x slowdown；`--cuda-sync` 会进一步串行化 CPU/GPU
- 与 `pdb` / `debugpy` 等同样使用 `sys.settrace` 的工具不能并存
- DDP 多 rank 训练时，每个 rank 都会写报告，请加 RANK 区分输出目录
- 当 `--start-at`/`--stop-at`/`--profile-hits` 的 file 模式命中多文件时，"第 N 次"是跨文件累加
- stop 之后 `--start-at` 不会再次触发（防止反复跳转）
- `--exclude-from` 不支持 gitignore 的 `!` 否定语法（会跳过并 warn）

## License

MIT，见 [LICENSE](./LICENSE)。