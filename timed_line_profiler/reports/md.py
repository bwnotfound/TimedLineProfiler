"""Markdown 报告：相对完整的聚合数据。

包含：
  - 概览（总耗时/命中数/文件数/窗口数）
  - 各文件聚合（每文件全行，按行号升序）
  - 选中行（max(top-k, ratio) 并集）的 bucket 分布矩阵
"""

import linecache
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from ..profiler import TimedLineProfiler
from ..selection import select_keys_per_file


def _esc(code: str, max_len: int = 120) -> str:
    """转义 markdown 表格单元格内容。"""
    c = code.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'")
    return c[:max_len]


def render_markdown(
    profiler: TimedLineProfiler,
    out_path: str,
    top_k: int = 10,
    top_ratio_pct: float = 0.0,
    threshold_ms: float = 0.0,
):
    agg = profiler.aggregate()
    lines: List[str] = ["# TimedLineProfiler 报告", ""]

    if not agg:
        lines.append("> 没有收集到任何性能数据。")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[ok] Markdown 报告已写入: {out_path}", file=sys.stderr)
        return

    bs = profiler.bucket_seconds
    bucket_count = profiler.max_bucket + 1
    total_t = sum(t for t, _ in agg.values())
    total_ms = total_t * 1000
    total_calls = sum(c for _, c in agg.values())
    files_set = {fn for fn, _ in agg.keys()}

    # ---- 概览 ----
    lines.append("## 概览")
    lines.append("")
    lines.append(f"- 总耗时: **{total_ms:.2f} ms** ({total_t:.6f} s)")
    lines.append(f"- 总命中次数: {total_calls}")
    lines.append(f"- 命中行数（去重）: {len(agg)}")
    lines.append(f"- 涉及文件数: {len(files_set)}")
    lines.append(f"- 时间窗口大小: {bs} s")
    lines.append(f"- 时间窗口数: {bucket_count}")
    lines.append(f"- 选行规则: 每文件 max(top-{top_k}, 占比 ≥ {top_ratio_pct}%)")
    lines.append(f"- 阈值过滤: 行总耗时 < {threshold_ms} ms 的不展示")
    lines.append("")

    # ---- 各文件聚合（全行）----
    files_data: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for (fn, ln), (t, c) in agg.items():
        files_data[fn].append((ln, t, c))
        file_total[fn] += t

    lines.append("## 各文件聚合（每文件全行，按行号升序）")
    lines.append("")
    for fn in sorted(files_data, key=lambda x: -file_total[x]):
        f_total_ms = file_total[fn] * 1000
        f_pct = f_total_ms / total_ms * 100 if total_ms else 0.0
        f_calls = sum(c for _, _, c in files_data[fn])
        lines.append(f"### `{fn}`")
        lines.append("")
        lines.append(
            f"文件总耗时: **{f_total_ms:.2f} ms** ({f_pct:.1f}%) · "
            f"命中次数: {f_calls} · 命中行数: {len(files_data[fn])}"
        )
        lines.append("")

        rows = sorted(files_data[fn])
        kept = [(ln, t, c) for ln, t, c in rows if t * 1000 >= threshold_ms]
        if not kept:
            lines.append("_（所有行都被阈值过滤）_")
            lines.append("")
            continue

        lines.append(
            "| 行 | 总耗时(ms) | 次数 | 平均(ms) | 文件内占比 | 全局占比 | 代码 |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for ln, t, c in kept:
            ms = t * 1000
            avg = ms / c if c else 0.0
            file_pct = ms / f_total_ms * 100 if f_total_ms else 0.0
            global_pct = ms / total_ms * 100 if total_ms else 0.0
            code = _esc(linecache.getline(fn, ln).strip())
            lines.append(
                f"| {ln} | {ms:.2f} | {c} | {avg:.4f} | "
                f"{file_pct:.1f}% | {global_pct:.1f}% | `{code}` |"
            )
        lines.append("")

    # ---- 函数级耗时 ----
    func_agg = profiler.aggregate_funcs()
    if func_agg:
        funcs_by_file: Dict[str, List[Tuple[str, int, float, int]]] = defaultdict(list)
        for (fn, fname, fl), (t, c) in func_agg.items():
            funcs_by_file[fn].append((fname, fl, t, c))

        lines.append("## 函数级耗时（含子调用；yield 之间等待时间不计入）")
        lines.append("")
        lines.append(
            "_call_count 是函数被进入的次数；generator 每次 next()/send() "
            "都算一次（创建生成器对象本身也算一次）_"
        )
        lines.append("")
        for fn in sorted(funcs_by_file, key=lambda x: -file_total.get(x, 0)):
            funcs_sorted = sorted(funcs_by_file[fn], key=lambda x: -x[2])
            funcs_kept = [f for f in funcs_sorted if f[2] * 1000 >= threshold_ms]
            if not funcs_kept:
                continue
            lines.append(f"### `{fn}`")
            lines.append("")
            lines.append(
                "| 函数 | def 行 | 总耗时(ms) | 调用次数 | 平均(ms) | 文件内占比 |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            f_total_ms = file_total[fn] * 1000
            for fname, fl, t, c in funcs_kept:
                ms = t * 1000
                avg = ms / c if c else 0.0
                file_pct = ms / f_total_ms * 100 if f_total_ms else 0.0
                lines.append(
                    f"| `{fname}` | {fl} | {ms:.2f} | {c} | {avg:.4f} | "
                    f"{file_pct:.1f}% |"
                )
            lines.append("")

    # ---- 选中行的 bucket 分布 ----
    selected_keys = select_keys_per_file(agg, top_k, top_ratio_pct)
    if bucket_count >= 1 and selected_keys:
        bucket_labels = [f"{i*bs:.1f}-{(i+1)*bs:.1f}s" for i in range(bucket_count)]
        matrix = profiler.per_bucket_matrix(selected_keys)
        lines.append(f"## 选中 {len(selected_keys)} 行的时间窗口分布（单位 ms）")
        lines.append("")
        lines.append(
            f"_选行规则：每文件 max(top-{top_k}, 占比 ≥ {top_ratio_pct}%)，"
            f"按全局耗时降序展示_"
        )
        lines.append("")
        header = "| 文件:行 | 总耗时(ms) | " + " | ".join(bucket_labels) + " |"
        sep_row = "|---|---:|" + "|".join(["---:"] * bucket_count) + "|"
        lines.append(header)
        lines.append(sep_row)
        import os

        for i, k in enumerate(selected_keys):
            fn, ln = k
            t = agg[k][0]
            row_label = f"`{os.path.basename(fn)}:{ln}`"
            cells = " | ".join(f"{matrix[i][j]:.2f}" for j in range(bucket_count))
            lines.append(f"| {row_label} | {t*1000:.2f} | {cells} |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] Markdown 报告已写入: {out_path}", file=sys.stderr)
