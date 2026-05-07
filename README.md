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
| **函数级耗时（含 yield 正确处理）**                | ❌             | ✅                   |
| **多线程支持（per-thread 数据）**                  | ❌             | ✅                   |

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

### LOCATOR 三种定位方式

`--start-at` / `--stop-at` 接受 `FILE:LOCATOR[:N]` 形式，LOCATOR 有三种：

| LOCATOR 形式     | 含义                                                                        | 例子                          |
| ---------------- | --------------------------------------------------------------------------- | ----------------------------- |
| `LINE`           | 绝对行号                                                                    | `train.py:120`                |
| `@FUNC[+OFFSET]` | 函数 def 行 + offset（offset≥1 才能稳定触发，因 def 行本身没有 line event） | `loop.py:@step+2`             |
| `~PATTERN`       | 用 regex 匹配该文件源码某一行                                               | `loop.py:~^\s*loss\.backward` |

`:N` 后缀（必须末尾纯数字）表示第 N 次匹配；缺省 N=1。N 解析规则：rsplit `:` 后**末尾纯数字段**且 rest 仍含 `:` 时才认作 N，否则末尾段是 LOCATOR。这意味着：

- `file.py:120` → LINE=120, N=1（兼容旧语法）
- `file.py:120:5` → LINE=120, N=5（兼容旧语法）
- `file.py:~hello:world` → PATTERN=`hello:world`, N=1（末尾 `world` 非数字，整体当 PATTERN）
- `file.py:~run\d+:5` → PATTERN=`run\d+`, N=5（末尾 `5` 当 N；要 PATTERN 含末尾 `:数字` 请显式加 `:1`）

### 开始

- `--start-at FILE:LOCATOR[:N]`：上述三种方式之一触发开始记录
- 不指定 `--start-at` → 程序一启动就记录

### 停止

以下三者**任一先达到**就停止（OR 关系，"取 min"）：

- `--stop-at FILE:LOCATOR[:M]`：第 M 次命中（绝对计数，从程序启动起算）
- `--profile-hits M`：开始记录后，**`--start-at` 那行/那个匹配点**再命中 M 次时停止（相对计数）。需配合 `--start-at`
- `--max-duration SEC`：开始记录后最多 SEC 秒就停止

### 例子

```bash
# 行号模式（最常用）：profile 主循环 100 步
tlprof --target 'mymodel/**/*.py' \
    --start-at 'mymodel/trainer/loop.py:120' \
    --profile-hits 100 --out ./rep -- train.py

# 函数偏移模式：进入 step 函数体第 2 行就开始（不必查具体行号）
tlprof --target 'mymodel/**/*.py' \
    --start-at 'mymodel/trainer/loop.py:@step+2' \
    --profile-hits 100 --out ./rep -- train.py

# 正则模式：找含 loss.backward() 的那一行
tlprof --target 'mymodel/**/*.py' \
    --start-at 'mymodel/trainer/loop.py:~loss\.backward' \
    --profile-hits 100 --out ./rep -- train.py
```

> ⚠️ 当 trigger 涉及的 file 模式命中**多个文件**时，"第 N 次"是**全局累加**（跨文件计数）。要精确触发，请用具体单文件路径。

> ⚠️ stop 之后 `--start-at` **不会再次触发**（防止 stop → start → stop 反复跳转）。

> ⚠️ 函数偏移模式 `@FUNC+OFFSET`：offset 是相对该函数 def 行的偏移（def 行算 0），所以 offset=1 是 def 下面第 1 行，offset=2 是第 2 行……。注意 offset=0 即 def 行本身在 Python 中通常**不会**触发 line event，要稳定命中请用 ≥1。

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

# 下钻到单个线程：
for t in profiler.list_threads():
    agg_t = profiler.aggregate(thread=t['tid'])
    funcs_t = profiler.aggregate_funcs(thread=t['tid'])
    total_ms = sum(_t for _t, _ in agg_t.values()) * 1000
    print(f"{t['name']:20s} {total_ms:8.2f} ms  {len(agg_t)} 行")
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

| 参数                         | 说明                                                                  |
| ---------------------------- | --------------------------------------------------------------------- |
| `--target`                   | 目标文件；可重复，单个内可用逗号分隔；FS glob 优先 + 相对路径模式兜底 |
| `--exclude`                  | 排除文件 glob；可重复，单个内可用逗号分隔                             |
| `--exclude-from`             | gitignore 风格排除规则文件；可重复，单个内可用逗号分隔                |
| `--bucket`                   | 时间窗口大小（秒）                                                    |
| `--start-at FILE:LINE[:N]`   | 该行命中第 N 次时开始记录（N 默认 1）                                 |
| `--stop-at FILE:LINE[:M]`    | 该行命中第 M 次时停止记录（M 默认 1，绝对计数）                       |
| `--profile-hits M`           | 开始记录后 --start-at 那行再命中 M 次后停止（相对计数）               |
| `--max-duration SEC`         | 开始记录后最多 SEC 秒就停止；与上述 stop 条件 OR 关系                 |
| `--cuda-sync`                | 每个目标行后调 `torch.cuda.synchronize()`                             |
| `--out`                      | 输出目录（不存在时自动创建），生成 report.html + report.md            |
| `--top-k`                    | 每个文件至少选 k 行（默认 10）                                        |
| `--top-ratio PCT`            | 占文件总耗时 ≥ PCT% 的行也选上，与 top-k 取并集（默认 0）             |
| `--threshold-ms`             | 报告中过滤总耗时低于该值的行                                          |
| `--html-max-file-views N`    | HTML dropdown 中最多列出 top-N 个单文件视图（默认 8）                 |
| `--html-all-files`           | HTML dropdown 列出全部文件（覆盖 max-file-views，HTML 会变大）        |
| `--main-thread-only-trigger` | 只让主线程参与 trigger 计数；默认任意线程命中都算                     |

## 异常退出与崩溃排查

profiler 尽力在程序异常退出时仍输出已采集到的数据：

| 退出方式                     | 报告是否生成 | 备注                           |
| ---------------------------- | ------------ | ------------------------------ |
| 正常结束 / `sys.exit(N)`     | ✅            | exit code 透传                 |
| `KeyboardInterrupt` (Ctrl+C) | ✅            |                                |
| 训练脚本抛 Python 异常       | ✅            | 异常 traceback 也会打印        |
| 训练脚本被 `kill -9` / OOM   | ❌            | 进程被立刻终止，无任何机会落盘 |
| 训练脚本触发 C++ `abort()`   | ❌            | 同上，绕过 atexit              |

异常发生时，profiler 会**立刻关闭 `sys.settrace`**，避免在异常 unwind 路径（C++ 销毁器、`__del__`、teardown 等）上继续 trace。这能减少 trace 与 C 扩展（如 PyTorch / Lightning）异常清理代码相互干扰导致的次级崩溃。

**遇到崩溃如何排查**

如果训练在 lightning / pytorch 内部崩溃（典型症状如 `RuntimeError: unknown parameter type`、`pure virtual method called` 之类），**先做对照实验**：把同样的命令去掉 `tlprof` 部分直接跑，看是否同样崩溃。

- 同样崩溃 → 与 profiler 无关，是训练环境本身的兼容性问题（lightning vs torch 版本、optimizer state 类型等），需要在不带 profiler 的环境里先解决
- 仅 tlprof 下崩溃 → 是 profiler 的 trace 干扰，请提交 issue 并附上完整 traceback

## 函数级耗时统计

除了行级数据，每份报告（text / md）的每个文件后会列出该文件中**所有函数的总耗时**。

- **总耗时含子调用**：与 cProfile 的 cumulative time 一致——`A` 调用 `B`，`A` 的耗时含 `B` 的全部时间
- **generator / yield 自动正确处理**：`sys.settrace` 把 yield 视为一次 return（值带在 arg 里），下次 `next()` 时再触发 call。两段时间分别累加，**yield 之间消费者的等待时间不会被计入 generator**
- `call_count` 是函数被进入的次数。注意 generator 函数：**创建 generator 对象本身**也算一次 call/return（耗时近 0），所以 `call_count = 1（创建）+ N（实际 next() 次数）`
- 数据生成机制：在 `'call'` 事件记录 frame 进入时间，`'return'` 事件累加 `now - max(call_time, recording_start)` 到该函数

例子（已在 `examples/` 中给出）：caller 在 yield 之间 sleep 1.5s 不会被算到 generator 函数上，generator 仅记录它自己 yield 段的真实工作时间。

## 多线程支持

Python 同进程内的子线程（`threading.Thread` / `ThreadPoolExecutor`）会被自动 trace。每个线程的数据独立采集（**无锁**——每个线程只写自己的 dict），最后报告里既可以看合并视图，也可以下钻到单个线程。

### 工作机制

`profiler.start()` 调用 `threading.settrace(_global_trace)`——这之后**新创建**的所有 Python 线程会自动被 trace。已存在的线程不受影响（但启动 profiler 时通常只有主线程，所以无所谓）。

数据结构：

```python
profiler._thread_state[tid] = {
    'last_line', 'last_time',           # per-thread 状态，互不干扰
    'bucket_data', 'func_bucket_data',  # per-thread 累加
    'meta': {
        'name', 'is_main', 'order',
        'first_seen_perf', 'first_seen_wall',  # 相对+绝对时间戳
        'last_seen_perf',  'last_seen_wall',
    },
}
```

每条数据的 thread_id 始终独立，互相不踩。

### 报告呈现

text / md 报告：

- 顶部"线程总览"列出每个线程的活跃时段、命中行数、命中次数、耗时
- 文件/行/函数表是**合并视图**（所有线程数据加在一起）

下钻到单线程的代码 API：

```python
# 通过 Python API（不用 cli），可以拿任意线程的独立数据：
agg_main = profiler.aggregate(thread='main')        # 仅主线程
agg_t    = profiler.aggregate(thread=tid)           # 指定线程
agg_all  = profiler.aggregate()                     # 合并（默认）

func_main = profiler.aggregate_funcs(thread='main')
threads   = profiler.list_threads()                 # 所有线程的 meta 信息
```

> HTML 报告中的分线程下钻（dropdown 切换）和时间轴可视化（gantt-style 显示每线程的活跃区间，支持相对/绝对时间 toggle）将在后续版本中加入。

### Trigger 在多线程下的语义

- **默认**：start-at / stop-at / profile-hits 的 "N 次命中" 是**任何线程命中都算**（profiler-wide 计数）
- **`--main-thread-only-trigger`**：只主线程命中才算。常用场景：你想精确控制 profile 主循环的第 N 步，避免 worker 线程干扰计数

无论选哪种，trigger 一旦触发，**所有线程**都进入 recording 状态（recording 是 profiler 全局状态）。

### 边界

- **子进程不被支持**（如 PyTorch DataLoader workers / multiprocessing）。`sys.settrace` / `threading.settrace` 都跨不过进程边界。如果你的工作负载主要在子进程里，profiler 看不到
- **C 扩展直接起的线程**（如 NumPy 内部的某些 OpenMP 池）也不受 `threading.settrace` 影响——只有用 Python `threading` 模块创建的线程会被 hook
- **trace 已经存在的线程**没法注入。但 profiler 一般在 main 入口启动，那时候只有主线程

## 已知局限

- 子进程（如 DataLoader workers、multiprocessing、DDP 不同 rank）不会被追踪——`sys.settrace` 跨不过进程边界。同进程内的子线程是支持的，详见"多线程支持"
- `sys.settrace` 自带 5–30x slowdown；`--cuda-sync` 会进一步串行化 CPU/GPU
- 与 `pdb` / `debugpy` 等同样使用 `sys.settrace` 的工具不能并存
- DDP 多 rank 训练时，每个 rank 都会写报告，请加 RANK 区分输出目录
- 当 `--start-at`/`--stop-at`/`--profile-hits` 的 file 模式命中多文件时，"第 N 次"是跨文件累加
- stop 之后 `--start-at` 不会再次触发（防止反复跳转）
- `--exclude-from` 不支持 gitignore 的 `!` 否定语法（会跳过并 warn）
- **耗时归属语义**：每行的耗时 = "该行 line 事件触发 → 下一次 line 事件触发"。这意味着：
  - 若该行调用了 **target 文件外** 的代码（如调用 lightning/torch 内部），那段时间会被算到该调用所在行（这通常是用户想看到的——能定位"哪行调用了什么导致它慢"）。
  - 函数末尾的 `return` 行耗时精确到 return 完成的瞬间（不会被错误延伸到 caller 的下一段代码）。
  - 函数 **第一行** 的耗时包含了 Python 函数调用本身的开销（call 协议 + 局部变量初始化等）。

## License

MIT，见 [LICENSE](./LICENSE)。