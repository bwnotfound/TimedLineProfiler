"""Markdown 报告：线程总览（活跃率 + 折叠 silent）+ 三个口径 + 函数 self/cumul + 每线程下钻。"""

import linecache
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..profiler import TimedLineProfiler
from ..selection import select_keys_per_file


def _esc(code: str, max_len: int = 120) -> str:
    """转义 markdown 表格单元格内容。"""
    c = code.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'")
    return c[:max_len]


def _infer_thread_alias(profiler: TimedLineProfiler, tid: int) -> Optional[str]:
    """该线程命中最多的函数名作为 alias hint。"""
    funcs = profiler.aggregate_funcs(thread=tid)
    if not funcs:
        return None
    top = max(funcs.items(), key=lambda x: x[1][2])  # by count
    (_fn, fname, _fl), (_cumul, _self, count) = top
    return fname if count > 0 else None


def render_markdown(
    profiler: TimedLineProfiler,
    out_path: str,
    top_k: int = 10,
    top_ratio_pct: float = 0.0,
    threshold_ms: float = 0.0,
    per_thread_top_lines: int = 10,
    per_thread_top_funcs: int = 5,
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

    rec_dur = profiler.recording_duration
    rec_dur_ms = rec_dur * 1000 if rec_dur is not None else 0.0
    main_agg = profiler.aggregate(thread="main")
    main_total_ms = sum(t for t, _ in main_agg.values()) * 1000

    # ---- 概览：三个口径 ----
    lines.append("## 概览")
    lines.append("")
    lines.append("**三个耗时口径**（含义不同，看清楚再判断优化目标）：")
    lines.append("")
    lines.append(
        f"- **wall-clock 时长**: `{rec_dur_ms:.2f} ms` —— 程序真实流逝时间（profiler.start → stop）"
    )
    lines.append(
        f"- **主线程累加耗时**: `{main_total_ms:.2f} ms` —— 仅主线程命中的行 sum"
    )
    lines.append(
        f"- **合并累加耗时**: `{total_ms:.2f} ms` —— 所有线程命中的行 sum（≈ wall-clock × 活跃线程数）"
    )
    lines.append("")
    lines.append(f"其它统计：")
    lines.append("")
    lines.append(f"- 总命中次数: {total_calls}")
    lines.append(f"- 命中行数（去重）: {len(agg)}")
    lines.append(f"- 涉及文件数: {len(files_set)}")
    lines.append(f"- 时间窗口大小: {bs} s · 窗口数: {bucket_count}")
    lines.append(f"- 选行规则: 每文件 max(top-{top_k}, 占比 ≥ {top_ratio_pct}%)")
    lines.append(f"- 阈值过滤: 行总耗时 < {threshold_ms} ms 的不展示")
    lines.append("")

    # ---- 线程总览 ----
    threads = profiler.list_threads()
    active_threads = []
    silent_threads = []
    for t in threads:
        agg_t = profiler.aggregate(thread=t["tid"])
        if agg_t:
            t_total = sum(_t for _t, _ in agg_t.values())
            t_hits = sum(c for _, c in agg_t.values())
            duration = t["last_seen_perf"] - t["first_seen_perf"]
            t["_total_s"] = t_total
            t["_hits"] = t_hits
            t["_duration_s"] = duration
            t["_lines"] = len(agg_t)
            t["_alias"] = _infer_thread_alias(profiler, t["tid"])
            active_threads.append(t)
        else:
            silent_threads.append(t)

    if active_threads:
        lines.append("## 线程总览")
        lines.append("")
        lines.append(
            f"_共 {len(active_threads)} 个有命中线程"
            + (
                f"；另有 {len(silent_threads)} 个被注册但 0 hits 的线程已折叠到末尾_"
                if silent_threads
                else "_"
            )
        )
        lines.append("")
        lines.append(
            "| 线程 | 主 | 活跃率 | 活跃时段(ms) | 命中行 | 命中次数 | 耗时(ms) | 占合并 | hint |"
        )
        lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|---|")
        for t in active_threads:
            main_tag = "✓" if t["is_main"] else ""
            duration_ms = t["_duration_s"] * 1000
            t_total_ms = t["_total_s"] * 1000
            pct = t_total_ms / total_ms * 100 if total_ms else 0.0
            if t["_duration_s"] > 0:
                activity_pct = t["_total_s"] / t["_duration_s"] * 100
                activity_str = f"{activity_pct:.1f}%"
            else:
                activity_str = "—"
            alias_str = f"`{t['_alias']}`" if t["_alias"] else ""
            lines.append(
                f"| `{t['name']}` | {main_tag} | {activity_str} | "
                f"{duration_ms:.2f} | {t['_lines']} | {t['_hits']} | "
                f"{t_total_ms:.2f} | {pct:.1f}% | {alias_str} |"
            )
        lines.append("")
        lines.append(
            "> **活跃率** = 累加耗时 / 活跃时段；>100% 不会出现（一个线程内同一时刻只跑一行），"
            "<100% 表示该线程有 idle/wait 比例。**hint** 是该线程命中最多 hits 的函数名，仅供识别。"
        )
        lines.append("")

    # ---- 各文件聚合（合并视图，行级）----
    files_data: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for (fn, ln), (t, c) in agg.items():
        files_data[fn].append((ln, t, c))
        file_total[fn] += t

    lines.append("## 各文件聚合（合并视图，每文件全行）")
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

    # ---- 函数级耗时（self/cumul + ⓖ）----
    func_agg = profiler.aggregate_funcs()
    if func_agg:
        funcs_by_file: Dict[str, List[Tuple[str, int, float, float, int]]] = (
            defaultdict(list)
        )
        for (fn, fname, fl), (cumul, self_t, cnt) in func_agg.items():
            funcs_by_file[fn].append((fname, fl, cumul, self_t, cnt))

        lines.append("## 函数级耗时")
        lines.append("")
        lines.append(
            "- **self**：函数自己代码的耗时（**不含**子调用）。占文件比 = self / 文件行级 sum，永远 ≤ 100%"
        )
        lines.append(
            "- **cumul**：函数 call→return 的总耗时（**含**子调用）。可能远超 self（嵌套 / 多层调用）"
        )
        lines.append(
            "- **进入次数**：generator 每次 next()/send() 都算一次（创建对象本身也算一次）"
        )
        lines.append("- **ⓖ**：generator 函数（`co_flags & 0x20`）")
        lines.append("")
        for fn in sorted(funcs_by_file, key=lambda x: -file_total.get(x, 0)):
            funcs_sorted = sorted(funcs_by_file[fn], key=lambda x: -x[3])  # by self
            funcs_kept = [
                f
                for f in funcs_sorted
                if f[2] * 1000 >= threshold_ms or f[3] * 1000 >= threshold_ms
            ]
            if not funcs_kept:
                continue
            lines.append(f"### `{fn}`")
            lines.append("")
            lines.append(
                "| 函数 | def 行 | self(ms) | cumul(ms) | 进入次数 | self 平均(ms) | self/文件 |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            f_total_ms = file_total.get(fn, 0) * 1000
            for fname, fl, cumul, self_t, cnt in funcs_kept:
                self_ms = self_t * 1000
                cumul_ms = cumul * 1000
                avg_self = self_ms / cnt if cnt else 0.0
                file_pct = self_ms / f_total_ms * 100 if f_total_ms else 0.0
                gen_tag = "ⓖ " if profiler.is_generator_func(fn, fl) else ""
                lines.append(
                    f"| {gen_tag}`{fname}` | {fl} | {self_ms:.2f} | {cumul_ms:.2f} | "
                    f"{cnt} | {avg_self:.4f} | {file_pct:.1f}% |"
                )
            lines.append("")

    # ---- 每线程下钻 ----
    sub_threads = [t for t in active_threads if not t["is_main"]]
    if sub_threads:
        lines.append(
            f"## 每线程下钻（top-{per_thread_top_lines} 行 + top-{per_thread_top_funcs} 函数）"
        )
        lines.append("")
        for t in sub_threads:
            tid = t["tid"]
            alias = f" — `{t['_alias']}`" if t["_alias"] else ""
            lines.append(f"### `{t['name']}`{alias}")
            lines.append("")
            lines.append(
                f"耗时 **{t['_total_s']*1000:.2f} ms** · "
                f"命中 {t['_hits']} 次 · "
                f"活跃时段 {t['_duration_s']*1000:.2f} ms · "
                f"活跃率 "
                f"{(t['_total_s']/t['_duration_s']*100):.1f}%"
                if t["_duration_s"] > 0
                else f"耗时 **{t['_total_s']*1000:.2f} ms** · " f"命中 {t['_hits']} 次"
            )
            lines.append("")
            agg_t = profiler.aggregate(thread=tid)
            top_lines = sorted(agg_t.items(), key=lambda x: -x[1][0])[
                :per_thread_top_lines
            ]
            if top_lines:
                lines.append(f"**Top {len(top_lines)} 行：**")
                lines.append("")
                lines.append("| 文件:行 | 总耗时(ms) | 次数 | 平均(ms) | 代码 |")
                lines.append("|---|---:|---:|---:|---|")
                for (lfn, ln), (lt, lc) in top_lines:
                    code = _esc(linecache.getline(lfn, ln).strip(), 80)
                    avg = lt * 1000 / lc if lc else 0.0
                    lines.append(
                        f"| `{os.path.basename(lfn)}:{ln}` | {lt*1000:.2f} | {lc} | "
                        f"{avg:.4f} | `{code}` |"
                    )
                lines.append("")
            funcs_t = profiler.aggregate_funcs(thread=tid)
            top_funcs = sorted(funcs_t.items(), key=lambda x: -x[1][1])[
                :per_thread_top_funcs
            ]
            if top_funcs:
                lines.append(f"**Top {len(top_funcs)} 函数（按 self 降序）：**")
                lines.append("")
                lines.append("| 函数 | self(ms) | cumul(ms) | 进入次数 |")
                lines.append("|---|---:|---:|---:|")
                for (ffn, fname, fl), (cumul, self_t, cnt) in top_funcs:
                    gen_tag = "ⓖ " if profiler.is_generator_func(ffn, fl) else ""
                    lines.append(
                        f"| {gen_tag}`{os.path.basename(ffn)}:{fname}` | "
                        f"{self_t*1000:.2f} | {cumul*1000:.2f} | {cnt} |"
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
        for i, k in enumerate(selected_keys):
            fn, ln = k
            t = agg[k][0]
            row_label = f"`{os.path.basename(fn)}:{ln}`"
            cells = " | ".join(f"{matrix[i][j]:.2f}" for j in range(bucket_count))
            lines.append(f"| {row_label} | {t*1000:.2f} | {cells} |")
        lines.append("")

    # ---- 0 hits 线程列表（折叠到末尾）----
    if silent_threads:
        lines.append(f"## 附录：{len(silent_threads)} 个无命中线程")
        lines.append("")
        lines.append(
            "_这些线程被 trace 注册（有 'call' 事件）但没产生任何 line 数据；通常是 worker pool 中创建后未运行 target 代码的线程_"
        )
        lines.append("")
        lines.append("| 线程 | tid |")
        lines.append("|---|---:|")
        for t in silent_threads:
            lines.append(f"| `{t['name']}` | {t['tid']} |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] Markdown 报告已写入: {out_path}", file=sys.stderr)
