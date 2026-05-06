"""文本报告：兼容 LineProfiler 风格的每文件全行展示。"""

import linecache
from collections import defaultdict
from typing import Dict, List, Tuple

from ..profiler import TimedLineProfiler


def render_text(profiler: TimedLineProfiler, threshold_ms: float = 0.0) -> str:
    """生成 LineProfiler 风格的文本汇总（每文件全行，按行号排）。"""
    agg = profiler.aggregate()
    if not agg:
        return "[TimedLineProfiler] 没有收集到任何性能数据\n"

    files_data: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for (fn, ln), (t, c) in agg.items():
        files_data[fn].append((ln, t, c))
        file_total[fn] += t

    out: List[str] = []
    sep = "=" * 100
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
    return "\n".join(out) + "\n"
