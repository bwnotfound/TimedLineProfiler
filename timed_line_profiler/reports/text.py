"""文本报告：兼容 LineProfiler 风格的每文件全行展示。"""

import linecache
from collections import defaultdict
from typing import Dict, List, Tuple

from ..profiler import TimedLineProfiler


def render_text(profiler: TimedLineProfiler, threshold_ms: float = 0.0) -> str:
    """生成 LineProfiler 风格的文本汇总（每文件全行 + 函数级耗时表，按行号排）。

    顶部加"线程总览"。文件/行/函数数据是**所有线程合并**视图。
    """
    agg = profiler.aggregate()
    if not agg:
        return "[TimedLineProfiler] 没有收集到任何性能数据\n"

    out: List[str] = []
    sep = "=" * 100

    # ---- 线程总览 ----
    threads = profiler.list_threads()
    if threads:
        out.append(sep)
        out.append(f"线程总览：共 {len(threads)} 个线程被 trace（数据为合并视图）")
        out.append(sep)
        for t in threads:
            tid = t["tid"]
            agg_t = profiler.aggregate(thread=tid)
            t_total_ms = sum(_t for _t, _ in agg_t.values()) * 1000
            t_hits = sum(c for _, c in agg_t.values())
            t_lines = len(agg_t)
            duration_ms = (t["last_seen_perf"] - t["first_seen_perf"]) * 1000
            tag = " [main]" if t["is_main"] else ""
            out.append(
                f"  - {t['name']:30s} (tid={tid}){tag}  "
                f"活跃 {duration_ms:9.2f} ms  "
                f"命中行 {t_lines:4d}  命中次数 {t_hits:7d}  "
                f"耗时 {t_total_ms:9.2f} ms"
            )
        out.append("")

    files_data: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for (fn, ln), (t, c) in agg.items():
        files_data[fn].append((ln, t, c))
        file_total[fn] += t

    # 函数级数据按文件分组（合并视图）
    func_agg = profiler.aggregate_funcs()
    funcs_by_file: Dict[str, List[Tuple[str, int, float, int]]] = defaultdict(list)
    for (fn, fname, fl), (t, c) in func_agg.items():
        funcs_by_file[fn].append((fname, fl, t, c))

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
        out.append(
            f"{'-' * ln_w}-+-{'-' * time_w}-+-{'-' * 8}-+-{'-' * 10}-+{'-' * 60}"
        )

        d = {ln: (t, c) for ln, t, c in rows}
        for ln in range(rows[0][0], rows[-1][0] + 1):
            code = linecache.getline(fn, ln).rstrip("\n")
            if ln in d:
                t, c = d[ln]
                ms = t * 1000
                avg = ms / c if c else 0.0
                if ms >= threshold_ms:
                    out.append(
                        f"{ln:>{ln_w}} | {ms:>{time_w}.2f} | {c:>8} | "
                        f"{avg:>10.4f} | {code}"
                    )

        # 函数耗时小节（按 total 降序）
        funcs = funcs_by_file.get(fn)
        if funcs:
            funcs_sorted = sorted(funcs, key=lambda x: -x[2])
            max_fname_w = max(8, max(len(f[0]) for f in funcs_sorted))
            max_fms = max(f[2] * 1000 for f in funcs_sorted)
            fms_w = max(10, len(f"{max_fms:.2f}"))
            out.append("")
            out.append(
                f"  函数耗时（含子调用；yield 之间等待时间不计入）："
                f"{len(funcs_sorted)} 个函数"
            )
            out.append(
                f"  {'函数名'.ljust(max_fname_w)} | "
                f"{'def 行'.rjust(6)} | {'总耗时(ms)'.rjust(fms_w)} | "
                f"{'调用次数'.rjust(8)} | {'平均(ms)'.rjust(10)}"
            )
            out.append(
                f"  {'-' * max_fname_w}-+-{'-' * 6}-+-{'-' * fms_w}-+-"
                f"{'-' * 8}-+-{'-' * 10}"
            )
            for fname, fl, t, c in funcs_sorted:
                ms = t * 1000
                if ms < threshold_ms:
                    continue
                avg = ms / c if c else 0.0
                out.append(
                    f"  {fname.ljust(max_fname_w)} | {fl:>6} | "
                    f"{ms:>{fms_w}.2f} | {c:>8} | {avg:>10.4f}"
                )
    return "\n".join(out) + "\n"
