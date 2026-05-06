#!/usr/bin/env python3
"""
line_time_profiler.py - 带时间窗口的逐行性能分析器

特性：
    1. 不修改训练脚本，通过 runner 启动（runpy）
    2. 逐行追踪指定文件的执行耗时（基于 sys.settrace）
    3. 按时间窗口 (bucket) 聚合，可观察行耗时随训练进程的变化
    4. 支持 glob 通配符指定多个目标文件
    5. 起止触发器：按"某行命中第 N 次"开启/关闭记录，可跳过 import 与 warmup
    6. --cuda-sync：在每个目标行后调用 torch.cuda.synchronize()，使 GPU 行耗时
       反映真实端到端时间（默认关闭，关闭时测的是 CPU launch 时间）
    7. HTML 可视化（plotly）+ 文本报告（兼容 LineProfiler 风格）

用法示例：
    # 基础用法
    python line_time_profiler.py \
        --target demo_train.py \
        --bucket 0.5 --out report.html \
        -- demo_train.py

    # 完整用法（GPU 训练，跳过 5 次 warmup，profile 100 步，glob 多文件）
    python line_time_profiler.py \
        --target 'model/**/*.py' --target 'trainer/loop.py' \
        --bucket 5.0 --top-k 30 \
        --start-at trainer/loop.py:120:5 \
        --stop-at  trainer/loop.py:120:105 \
        --cuda-sync \
        --out report.html --txt report.txt \
        -- trainer/loop.py --epochs 100

已知局限：
    - 子进程（如 DataLoader workers）不会被追踪，只追踪主进程
    - sys.settrace 本身有 5-30x slowdown；--cuda-sync 会进一步串行化 CPU/GPU
    - 与 pdb / debugpy 等同样使用 trace 的工具不能并存
"""

import argparse
import atexit
import glob
import linecache
import os
import runpy
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _parse_trigger(spec: Optional[str]) -> Optional[Tuple[str, int, int]]:
    """解析 'file.py:line:n' -> (abspath, line, n)。"""
    if spec is None:
        return None
    parts = spec.rsplit(':', 2)
    if len(parts) != 3:
        raise ValueError(f"trigger 格式应为 FILE:LINE:N，实际收到: {spec}")
    fn, ln, n = parts
    return (os.path.abspath(fn), int(ln), int(n))


def _expand_targets(patterns: List[str]) -> Set[str]:
    """展开 glob 模式，返回绝对路径集合。"""
    files: Set[str] = set()
    for pat in patterns:
        # ** 需要 recursive=True
        matched = glob.glob(pat, recursive=True)
        if not matched:
            if os.path.exists(pat):
                files.add(os.path.abspath(pat))
            else:
                print(f"[warn] 未匹配到任何文件: {pat}", file=sys.stderr)
            continue
        for m in matched:
            if os.path.isfile(m) and m.endswith('.py'):
                files.add(os.path.abspath(m))
    return files


# ---------------------------------------------------------------------------
# 核心 Profiler
# ---------------------------------------------------------------------------

class TimedLineProfiler:
    def __init__(
        self,
        target_files: Set[str],
        bucket_seconds: float = 5.0,
        start_trigger: Optional[Tuple[str, int, int]] = None,
        stop_trigger: Optional[Tuple[str, int, int]] = None,
        cuda_sync: bool = False,
    ):
        self.target_files: Set[str] = set(target_files)
        self.bucket_seconds = bucket_seconds
        self.start_trigger = start_trigger
        self.stop_trigger = stop_trigger
        self.start_hit = 0
        self.stop_hit = 0
        # 没有 start_trigger 就立即开始记录
        self.recording = (start_trigger is None)

        # bucket_data[bucket_idx][(filename, lineno)] = [total_time, count]
        self.bucket_data: Dict[int, Dict[Tuple[str, int], List]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0])
        )

        self.start_wall: Optional[float] = None  # recording 真正开始的 perf_counter
        self.last_line: Optional[Tuple[str, int]] = None
        self.last_time: Optional[float] = None
        self.previous_trace = None
        self.enabled = False
        self.max_bucket = 0
        # raw co_filename -> abspath（若是 target）or None（若不是 target）
        self._fname_cache: Dict[str, Optional[str]] = {}

        # CUDA 同步
        self._cuda_sync_fn = None
        if cuda_sync:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[warn] --cuda-sync 已开启但 CUDA 不可用，将跳过同步",
                          file=sys.stderr)
                else:
                    self._cuda_sync_fn = torch.cuda.synchronize
                    print("[info] --cuda-sync 已开启：每个目标行后将调用 "
                          "torch.cuda.synchronize()", file=sys.stderr)
            except ImportError:
                print("[warn] --cuda-sync 已开启但未安装 torch，将跳过同步",
                      file=sys.stderr)

    # ---- trace 钩子 -------------------------------------------------------

    def _resolve(self, fn: str) -> Optional[str]:
        """raw filename -> abspath（命中目标）或 None（非目标），带缓存。"""
        cached = self._fname_cache.get(fn, ...)
        if cached is not ...:
            return cached
        abs_fn = os.path.abspath(fn)
        result = abs_fn if abs_fn in self.target_files else None
        self._fname_cache[fn] = result
        return result

    def _global_trace(self, frame, event, arg):
        """全局 trace：'call' 事件决定是否进入 frame 的 line trace。"""
        if event != 'call':
            return None
        if self._resolve(frame.f_code.co_filename) is None:
            return None
        return self._local_trace

    def _local_trace(self, frame, event, arg):
        if event != 'line':
            return self._local_trace

        # frame 一定是 target 文件
        fn = self._fname_cache.get(frame.f_code.co_filename)
        if fn is None:  # 防御
            return self._local_trace
        key = (fn, frame.f_lineno)

        # ---- 起始触发器：未在记录时只看 trigger ----
        if not self.recording:
            if self.start_trigger is not None:
                t_fn, t_ln, t_n = self.start_trigger
                if key == (t_fn, t_ln):
                    self.start_hit += 1
                    if self.start_hit >= t_n:
                        # 清空 pending GPU 操作，避免污染后续测量
                        if self._cuda_sync_fn is not None:
                            self._cuda_sync_fn()
                        self.recording = True
                        self.start_wall = time.perf_counter()
                        self.last_line = None
                        self.last_time = self.start_wall
                        print(f"[info] 触发开始记录: {t_fn}:{t_ln} 第 {self.start_hit} 次",
                              file=sys.stderr)
            return self._local_trace

        # ---- recording 中：每行后同步 GPU（可选） ----
        if self._cuda_sync_fn is not None:
            self._cuda_sync_fn()

        now = time.perf_counter()

        # 把 last_line 这一行的耗时记到对应 bucket
        if self.last_line is not None and self.last_time is not None:
            elapsed = now - self.last_time
            bucket = int((self.last_time - self.start_wall) / self.bucket_seconds)
            if bucket < 0:
                bucket = 0
            if bucket > self.max_bucket:
                self.max_bucket = bucket
            slot = self.bucket_data[bucket][self.last_line]
            slot[0] += elapsed
            slot[1] += 1

        self.last_line = key
        self.last_time = now

        # ---- 停止触发器 ----
        if self.stop_trigger is not None:
            t_fn, t_ln, t_n = self.stop_trigger
            if key == (t_fn, t_ln):
                self.stop_hit += 1
                if self.stop_hit >= t_n:
                    self.recording = False
                    self.last_line = None
                    print(f"[info] 触发停止记录: {t_fn}:{t_ln} 第 {self.stop_hit} 次",
                          file=sys.stderr)

        return self._local_trace

    # ---- 生命周期 ---------------------------------------------------------

    def start(self):
        # 预填充缓存（target 自身映射到自己）
        for f in self.target_files:
            self._fname_cache[f] = f
        # 没有 start_trigger 的话立即开始记录，需要把 start_wall 初始化
        if self.recording:
            self.start_wall = time.perf_counter()
            self.last_time = self.start_wall
            self.last_line = None
        self.previous_trace = sys.gettrace()
        sys.settrace(self._global_trace)
        self.enabled = True

    def stop(self):
        if not self.enabled:
            return
        # 收尾：把最后一行的耗时也记上
        if (self.recording
                and self.last_line is not None
                and self.last_time is not None
                and self.start_wall is not None):
            if self._cuda_sync_fn is not None:
                self._cuda_sync_fn()
            now = time.perf_counter()
            elapsed = now - self.last_time
            bucket = int((self.last_time - self.start_wall) / self.bucket_seconds)
            if bucket < 0:
                bucket = 0
            if bucket > self.max_bucket:
                self.max_bucket = bucket
            slot = self.bucket_data[bucket][self.last_line]
            slot[0] += elapsed
            slot[1] += 1
        sys.settrace(self.previous_trace)
        self.enabled = False

    # ---- 数据导出 ---------------------------------------------------------

    def aggregate(self) -> Dict[Tuple[str, int], List]:
        agg: Dict[Tuple[str, int], List] = defaultdict(lambda: [0.0, 0])
        for bucket in self.bucket_data.values():
            for k, (t, c) in bucket.items():
                agg[k][0] += t
                agg[k][1] += c
        return agg

    def per_bucket_matrix(self, keys: List[Tuple[str, int]]) -> List[List[float]]:
        """返回 [len(keys)][num_buckets] 的耗时矩阵，单位毫秒。"""
        n_b = self.max_bucket + 1 if self.bucket_data else 1
        matrix = [[0.0] * n_b for _ in keys]
        idx = {k: i for i, k in enumerate(keys)}
        for b, bucket in self.bucket_data.items():
            for k, (t, _) in bucket.items():
                if k in idx:
                    matrix[idx[k]][b] = t * 1000.0
        return matrix


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def render_text(profiler: TimedLineProfiler, threshold_ms: float = 0.0) -> str:
    """生成 LineProfiler 风格的文本汇总。"""
    agg = profiler.aggregate()
    if not agg:
        return "[TimedLineProfiler] 没有收集到任何性能数据\n"

    files_data: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for (fn, ln), (t, c) in agg.items():
        files_data[fn].append((ln, t, c))
        file_total[fn] += t

    out: List[str] = []
    sep = '=' * 100
    for fn in sorted(files_data):
        out.append(sep)
        out.append(f"文件: {fn}")
        out.append(f"总耗时: {file_total[fn]*1000:.2f} ms ({file_total[fn]:.6f} s)")
        out.append(sep)

        rows = sorted(files_data[fn])
        max_ms = max(t * 1000 for _, t, _ in rows)
        time_w = max(8, len(f"{max_ms:.2f}"))
        ln_w = max(4, len(str(rows[-1][0])))
        out.append(
            f"{'行号'.rjust(ln_w)} | {'时间(ms)'.rjust(time_w)} | "
            f"{'次数'.rjust(8)} | {'平均(ms)'.rjust(10)} | 代码"
        )
        out.append(f"{'-' * ln_w}-+-{'-' * time_w}-+-{'-' * 8}-+-{'-' * 10}-+{'-' * 60}")

        d = {ln: (t, c) for ln, t, c in rows}
        for ln in range(rows[0][0], rows[-1][0] + 1):
            code = linecache.getline(fn, ln).rstrip('\n')
            if ln in d:
                t, c = d[ln]
                ms = t * 1000
                avg = ms / c if c else 0.0
                if ms >= threshold_ms:
                    out.append(
                        f"{ln:>{ln_w}} | {ms:>{time_w}.2f} | {c:>8} | "
                        f"{avg:>10.4f} | {code}"
                    )
    return '\n'.join(out) + '\n'


def render_html(profiler: TimedLineProfiler, out_path: str, top_k: int = 30):
    """plotly HTML 可视化：热力图 + Top-K 折线图 + 汇总表。"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("[warn] 未安装 plotly，跳过 HTML 报告。pip install plotly 可启用。",
              file=sys.stderr)
        return

    agg = profiler.aggregate()
    if not agg:
        print("[warn] 没有数据，跳过 HTML。", file=sys.stderr)
        return

    sorted_keys = sorted(agg.keys(), key=lambda k: agg[k][0], reverse=True)
    top_keys = sorted_keys[:top_k]

    bucket_count = profiler.max_bucket + 1
    bs = profiler.bucket_seconds
    bucket_labels = [f"{i*bs:.1f}-{(i+1)*bs:.1f}s" for i in range(bucket_count)]
    matrix_ms = profiler.per_bucket_matrix(top_keys)
    row_labels = [f"{os.path.basename(fn)}:{ln}" for fn, ln in top_keys]
    code_snips = [linecache.getline(fn, ln).strip()[:80] for fn, ln in top_keys]

    hover = [
        [
            f"行: {row_labels[i]}<br>窗口: {bucket_labels[j]}<br>"
            f"耗时: {matrix_ms[i][j]:.2f} ms<br>代码: {code_snips[i]}"
            for j in range(bucket_count)
        ]
        for i in range(len(top_keys))
    ]

    heatmap = go.Heatmap(
        z=matrix_ms, x=bucket_labels, y=row_labels,
        colorscale='Viridis',
        colorbar=dict(title='ms'),
        text=hover, hoverinfo='text',
    )

    show_n_lines = min(10, len(top_keys))
    line_traces = []
    for i in range(show_n_lines):
        line_traces.append(go.Scatter(
            x=bucket_labels, y=matrix_ms[i],
            mode='lines+markers',
            name=row_labels[i],
            hovertemplate=f"{row_labels[i]}<br>%{{x}}: %{{y:.2f}} ms<extra></extra>",
        ))

    total_ms = sum(t for t, _ in agg.values()) * 1000
    table_rows = []
    for k in top_keys:
        fn, ln = k
        t, c = agg[k]
        ms = t * 1000
        avg = ms / c if c else 0.0
        pct = ms / total_ms * 100 if total_ms else 0.0
        code = linecache.getline(fn, ln).strip()
        table_rows.append([
            f"{os.path.basename(fn)}:{ln}",
            f"{ms:.2f}", str(c), f"{avg:.4f}", f"{pct:.1f}%",
            code[:120],
        ])

    table = go.Table(
        header=dict(
            values=['行', '总耗时(ms)', '调用次数', '平均(ms)', '占比', '代码'],
            fill_color='#3a3a3a', font=dict(color='white'), align='left',
        ),
        cells=dict(
            values=list(zip(*table_rows)) if table_rows else [[] for _ in range(6)],
            align='left', font=dict(size=11), height=22,
        ),
    )

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.25, 0.35],
        subplot_titles=(
            f'热力图: Top-{len(top_keys)} 行 × 时间窗口（{bs}s/窗口）',
            f'折线图: Top-{show_n_lines} 行的耗时随时间变化',
            f'Top-{len(top_keys)} 行汇总',
        ),
        specs=[[{'type': 'heatmap'}], [{'type': 'scatter'}], [{'type': 'table'}]],
        vertical_spacing=0.08,
    )
    fig.add_trace(heatmap, row=1, col=1)
    for tr in line_traces:
        fig.add_trace(tr, row=2, col=1)
    fig.add_trace(table, row=3, col=1)

    fig.update_layout(
        title=f'TimedLineProfiler 报告（总耗时 {total_ms:.0f} ms, '
              f'{bucket_count} 个时间窗口）',
        height=1400, showlegend=True,
    )
    fig.update_xaxes(title_text='时间窗口', row=1, col=1)
    fig.update_yaxes(title_text='文件:行号', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title_text='时间窗口', row=2, col=1)
    fig.update_yaxes(title_text='耗时(ms)', row=2, col=1)

    pio.write_html(fig, out_path, include_plotlyjs='cdn')
    print(f"[ok] HTML 报告已写入: {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Line-level time profiler with bucketed time windows.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--target', action='append', required=True,
                        help='要追踪的文件，支持 glob 通配符（如 "model/**/*.py"），可重复指定')
    parser.add_argument('--bucket', type=float, default=5.0,
                        help='时间窗口大小（秒），默认 5.0')
    parser.add_argument('--start-at', default=None, metavar='FILE:LINE:N',
                        help='当 FILE 第 LINE 行被命中第 N 次时开始记录')
    parser.add_argument('--stop-at', default=None, metavar='FILE:LINE:M',
                        help='当 FILE 第 LINE 行被命中第 M 次时停止记录')
    parser.add_argument('--cuda-sync', action='store_true',
                        help='每个目标行后调用 torch.cuda.synchronize() 以测真实 GPU 耗时')
    parser.add_argument('--out', default='profile_report.html',
                        help='HTML 报告输出路径')
    parser.add_argument('--txt', default=None, help='文本报告输出路径（可选）')
    parser.add_argument('--top-k', type=int, default=30,
                        help='可视化中展示的 Top-K 行数，默认 30')
    parser.add_argument('--threshold-ms', type=float, default=0.0,
                        help='文本报告中过滤总耗时低于该值的行')
    parser.add_argument('script', help='要执行的训练脚本')
    parser.add_argument('script_args', nargs=argparse.REMAINDER,
                        help='传给训练脚本的参数')

    args = parser.parse_args()

    targets = _expand_targets(args.target)
    if not targets:
        print("[error] 没有匹配到任何目标文件，退出", file=sys.stderr)
        sys.exit(2)
    print(f"[info] 共 {len(targets)} 个目标文件被追踪：", file=sys.stderr)
    for t in sorted(targets):
        print(f"  - {t}", file=sys.stderr)

    start_trig = _parse_trigger(args.start_at)
    stop_trig = _parse_trigger(args.stop_at)
    if start_trig and start_trig[0] not in targets:
        print(f"[warn] --start-at 的文件 {start_trig[0]} 不在 --target 列表中，"
              f"该 trigger 将永远不会触发！", file=sys.stderr)
    if stop_trig and stop_trig[0] not in targets:
        print(f"[warn] --stop-at 的文件 {stop_trig[0]} 不在 --target 列表中，"
              f"该 trigger 将永远不会触发！", file=sys.stderr)

    profiler = TimedLineProfiler(
        target_files=targets,
        bucket_seconds=args.bucket,
        start_trigger=start_trig,
        stop_trigger=stop_trig,
        cuda_sync=args.cuda_sync,
    )

    finalize_done = [False]

    def finalize():
        if finalize_done[0]:
            return
        finalize_done[0] = True
        try:
            profiler.stop()
        finally:
            print("\n" + "=" * 100, file=sys.stderr)
            print("[TimedLineProfiler] 训练结束/中断，开始生成报告 ...",
                  file=sys.stderr)
            text = render_text(profiler, threshold_ms=args.threshold_ms)
            print(text)
            if args.txt:
                with open(args.txt, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"[ok] 文本报告已写入: {args.txt}", file=sys.stderr)
            render_html(profiler, args.out, top_k=args.top_k)

    atexit.register(finalize)

    sys.argv = [args.script] + (args.script_args or [])
    profiler.start()
    try:
        runpy.run_path(args.script, run_name='__main__')
    except KeyboardInterrupt:
        print("\n[info] 收到 KeyboardInterrupt，将生成报告 ...", file=sys.stderr)


if __name__ == '__main__':
    main()
