"""文本报告：兼容 LineProfiler 风格的每文件全行展示 + 线程下钻 + 函数 self/cumul。"""

import linecache
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..profiler import TimedLineProfiler


def _infer_thread_alias(profiler: TimedLineProfiler, tid: int) -> Optional[str]:
    """根据该线程命中最多 hits 的函数名作为 alias 提示。

    返回 None 如果该线程数据为空。
    """
    funcs = profiler.aggregate_funcs(thread=tid)
    if not funcs:
        return None
    top = max(funcs.items(), key=lambda x: x[1][2])  # by count
    (_fn, fname, _fl), (_cumul, _self, count) = top
    return f"{fname}" if count > 0 else None


def render_text(
    profiler: TimedLineProfiler,
    threshold_ms: float = 0.0,
    per_thread_top_lines: int = 10,
    per_thread_top_funcs: int = 5,
) -> str:
    """生成 LineProfiler 风格的文本汇总。

    顶部三个口径：wall-clock / 主线程累加 / 合并累加。
    线程总览：合并视图 + 每线程汇总，含活跃率（耗时/wall-clock 跨度）。
    主体：每文件全行 + 函数表（self/cumul + generator ⓖ）。
    线程下钻：对每个有 hits 的非主线程，列其 top-N 行 + top-K 函数。
    """
    agg = profiler.aggregate()
    if not agg:
        return "[TimedLineProfiler] 没有收集到任何性能数据\n"

    out: List[str] = []
    sep = "=" * 100

    # ---- 三个口径的总耗时 ----
    rec_dur = profiler.recording_duration
    rec_dur_ms = rec_dur * 1000 if rec_dur is not None else 0.0
    main_agg = profiler.aggregate(thread="main")
    main_total_ms = sum(t for t, _ in main_agg.values()) * 1000
    merged_total_ms = sum(t for t, _ in agg.values()) * 1000

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

    out.append(sep)
    out.append("耗时口径（三个数字含义不同，看清楚再判断优化目标）：")
    out.append(f"  wall-clock 时长 : {rec_dur_ms:>11.2f} ms   程序真实流逝时间")
    out.append(f"  主线程累加耗时  : {main_total_ms:>11.2f} ms   仅主线程命中的行 sum")
    out.append(
        f"  合并累加耗时    : {merged_total_ms:>11.2f} ms   所有线程命中的行 sum (≈ wall-clock × 活跃线程数)"
    )
    out.append("")
    out.append(
        f"线程总览：{len(active_threads)} 个有命中"
        + (
            f"（另有 {len(silent_threads)} 个被注册但 0 hits，已折叠）"
            if silent_threads
            else ""
        )
    )
    out.append(sep)
    out.append(
        f"  {'线程名':32s}  {'活跃率':>6s}  {'活跃时段':>11s}  "
        f"{'命中行':>5s}  {'命中次数':>8s}  {'耗时':>11s}  hint"
    )
    out.append(f"  {'-'*32}  {'-'*6}  {'-'*11}  {'-'*5}  {'-'*8}  {'-'*11}  ----")
    for t in active_threads:
        tag = " [main]" if t["is_main"] else ""
        if t["_duration_s"] > 0:
            activity_pct = t["_total_s"] / t["_duration_s"] * 100
            activity_str = f"{activity_pct:5.1f}%"
        else:
            activity_str = "  -- "
        alias_str = f"  [{t['_alias']}]" if t["_alias"] else ""
        out.append(
            f"  {(t['name']+tag)[:32]:32s}  {activity_str}  "
            f"{t['_duration_s']*1000:>9.2f} ms  "
            f"{t['_lines']:>5d}  {t['_hits']:>8d}  "
            f"{t['_total_s']*1000:>9.2f} ms{alias_str}"
        )
    if silent_threads:
        names_preview = ", ".join(t["name"] for t in silent_threads[:8])
        more = "..." if len(silent_threads) > 8 else ""
        out.append(f"  ({len(silent_threads)} 个 0 hits 线程：{names_preview}{more})")
    out.append("")

    # ---- 各文件聚合 ----
    files_data: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for (fn, ln), (t, c) in agg.items():
        files_data[fn].append((ln, t, c))
        file_total[fn] += t

    func_agg = profiler.aggregate_funcs()
    funcs_by_file: Dict[str, List[Tuple[str, int, float, float, int]]] = defaultdict(
        list
    )
    for (fn, fname, fl), (cumul, self_t, cnt) in func_agg.items():
        funcs_by_file[fn].append((fname, fl, cumul, self_t, cnt))

    for fn in sorted(files_data):
        out.append(sep)
        out.append(f"文件: {fn}")
        out.append(f"总耗时（合并视图行级 sum）: {file_total[fn]*1000:.2f} ms")
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

        # 函数表（self/cumul + ⓖ）
        funcs = funcs_by_file.get(fn)
        if funcs:
            funcs_sorted = sorted(funcs, key=lambda x: -x[3])  # by self
            max_fname_w = max(12, max(len(f[0]) for f in funcs_sorted) + 2)
            f_total_ms = file_total[fn] * 1000
            out.append("")
            out.append(
                f"  函数耗时（按 self 降序；ⓖ = generator；self/cumul 见 README）：{len(funcs_sorted)} 个"
            )
            out.append(
                f"  {'函数名'.ljust(max_fname_w)} | "
                f"{'def 行'.rjust(6)} | {'self(ms)'.rjust(10)} | "
                f"{'cumul(ms)'.rjust(10)} | "
                f"{'进入次数'.rjust(8)} | {'self/file'.rjust(10)}"
            )
            out.append(
                f"  {'-'*max_fname_w}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}"
            )
            for fname, fl, cumul, self_t, cnt in funcs_sorted:
                self_ms = self_t * 1000
                cumul_ms = cumul * 1000
                if self_ms < threshold_ms and cumul_ms < threshold_ms:
                    continue
                gen_tag = "ⓖ" if profiler.is_generator_func(fn, fl) else " "
                pct = self_ms / f_total_ms * 100 if f_total_ms else 0.0
                out.append(
                    f"  {gen_tag} {fname.ljust(max_fname_w-2)} | {fl:>6} | "
                    f"{self_ms:>10.2f} | {cumul_ms:>10.2f} | "
                    f"{cnt:>8} | {pct:>9.1f}%"
                )
        out.append("")

    # ---- 每线程独立 top-N（仅非主线程）----
    sub_threads = [t for t in active_threads if not t["is_main"]]
    if sub_threads:
        out.append(sep)
        out.append(
            f"每线程下钻（每线程 top-{per_thread_top_lines} 行 + top-{per_thread_top_funcs} 函数）"
        )
        out.append(sep)
        for t in sub_threads:
            tid = t["tid"]
            alias = f"  [{t['_alias']}]" if t["_alias"] else ""
            out.append("")
            out.append(
                f"--- {t['name']}{alias}  "
                f"({t['_total_s']*1000:.2f} ms / {t['_hits']} hits) ---"
            )
            agg_t = profiler.aggregate(thread=tid)
            top_lines = sorted(agg_t.items(), key=lambda x: -x[1][0])[
                :per_thread_top_lines
            ]
            if top_lines:
                out.append(f"  Top {len(top_lines)} 行：")
                for (lfn, ln), (lt, lc) in top_lines:
                    code = linecache.getline(lfn, ln).strip()[:80]
                    out.append(
                        f"    {os.path.basename(lfn):28s}:{ln:<5d}  "
                        f"{lt*1000:>9.2f} ms  × {lc:<6d}  {code}"
                    )
            funcs_t = profiler.aggregate_funcs(thread=tid)
            top_funcs = sorted(funcs_t.items(), key=lambda x: -x[1][1])[
                :per_thread_top_funcs
            ]
            if top_funcs:
                out.append(f"  Top {len(top_funcs)} 函数（按 self 降序）：")
                for (ffn, fname, fl), (cumul, self_t, cnt) in top_funcs:
                    gen_tag = "ⓖ" if profiler.is_generator_func(ffn, fl) else " "
                    out.append(
                        f"    {gen_tag} {os.path.basename(ffn):28s}:{fname:<25s} "
                        f"self {self_t*1000:>8.2f} ms  cumul {cumul*1000:>8.2f} ms  × {cnt}"
                    )

    return "\n".join(out) + "\n"
