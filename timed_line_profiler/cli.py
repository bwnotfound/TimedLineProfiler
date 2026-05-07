"""命令行入口：参数解析、目录预检查、Profiler 启动、报告生成。"""

import argparse
import atexit
import os
import runpy
import sys

from .pattern import (
    parse_trigger,
    resolve_excludes,
    resolve_targets,
)
from .profiler import TimedLineProfiler
from .reports import render_html, render_markdown, render_text


def _split_csv(values):
    """把 List[str] 中每项按逗号 split，flatten 后 strip + 去空。

    让 ``--target 'a,b'`` 等价于 ``--target a --target b``；两种形式可混用。
    不支持含逗号的文件名。
    """
    out = []
    for v in values or []:
        for item in v.split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Line-level time profiler with bucketed time windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help='要追踪的文件，支持 FS glob 与"相对路径 glob"两种语义；'
        "可重复指定，单个 --target 内也可用逗号分隔多个。"
        '例如 "model/**/*.py,trainer/loop.py"',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="排除文件的模式（与 --target 同 glob 语法）；"
        "可重复，单个内可用逗号分隔",
    )
    parser.add_argument(
        "--exclude-from",
        action="append",
        default=[],
        help="从 gitignore 风格文件读取排除规则；"
        "可重复，单个内可用逗号分隔多个文件路径。"
        "不支持 ! 否定语法（会跳过并 warn）",
    )
    parser.add_argument(
        "--bucket", type=float, default=5.0, help="时间窗口大小（秒），默认 5.0"
    )
    parser.add_argument(
        "--start-at",
        default=None,
        metavar="FILE:LINE[:N]",
        help="当 FILE 第 LINE 行被命中第 N 次时开始记录；"
        "N 默认 1（首次命中即开始）。"
        "FILE 同样支持 glob/相对路径",
    )
    parser.add_argument(
        "--stop-at",
        default=None,
        metavar="FILE:LINE[:M]",
        help="当 FILE 第 LINE 行被命中第 M 次时停止记录；"
        "M 默认 1。FILE 同样支持 glob/相对路径",
    )
    parser.add_argument(
        "--profile-hits",
        type=int,
        default=None,
        metavar="M",
        help="开始记录后，--start-at 那行被再命中 M 次时停止；"
        "需配合 --start-at 使用",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SEC",
        help="开始记录后最多运行 SEC 秒就停止；"
        "与 --profile-hits / --stop-at 同时指定时取先到先停（OR）",
    )
    parser.add_argument(
        "--cuda-sync",
        action="store_true",
        help="每个目标行后调用 torch.cuda.synchronize() 以测真实 GPU 耗时",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="报告输出目录（不存在时自动创建）；" "将生成 report.html 与 report.md",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="每个文件至少选 top-k 行展示在可视化中，默认 10",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.0,
        metavar="PCT",
        help="每行占该文件总耗时阈值（%%），占比 ≥ 该值的行也会被选中。"
        "与 --top-k 取并集。默认 0（仅 top-k 生效）",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=0.0,
        help="报告中过滤总耗时低于该值的行（ms）",
    )
    parser.add_argument(
        "--html-max-file-views",
        type=int,
        default=8,
        metavar="N",
        help="HTML dropdown 中最多列出 top-N 个单文件视图（按文件耗时降序）；"
        "默认 8。文件数很多时可降低 HTML 体积和加载时间。"
        "完整数据始终保留在 report.md 中",
    )
    parser.add_argument(
        "--html-all-files",
        action="store_true",
        help="HTML dropdown 中列出所有文件视图（覆盖 --html-max-file-views）；"
        "文件多时 HTML 会很大，仅在确实需要时开启",
    )
    parser.add_argument(
        "--main-thread-only-trigger",
        action="store_true",
        help="只让主线程的 line 事件参与 trigger 计数（start-at / "
        "stop-at / profile-hits）；默认任何线程命中都算",
    )
    parser.add_argument("script", help="要执行的训练脚本")
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="传给训练脚本的参数"
    )

    args = parser.parse_args()

    # 把可重复参数里的逗号分隔形式 flatten 出来
    args.target = _split_csv(args.target)
    args.exclude = _split_csv(args.exclude)
    args.exclude_from = _split_csv(args.exclude_from)

    # ---- 输出目录预检查 / 自动创建 ----
    if os.path.exists(args.out):
        if not os.path.isdir(args.out):
            print(
                f"[error] --out 指向的路径已存在但不是目录: {args.out}", file=sys.stderr
            )
            sys.exit(2)
    else:
        try:
            os.makedirs(args.out, exist_ok=True)
        except OSError as e:
            print(f"[error] 创建 --out 目录失败: {args.out} ({e})", file=sys.stderr)
            sys.exit(2)
        print(f"[info] 已创建输出目录: {args.out}", file=sys.stderr)

    # ---- target / exclude 解析 ----
    abs_targets, rel_patterns = resolve_targets(args.target)
    if not abs_targets and not rel_patterns:
        print("[error] 没有任何 --target，退出", file=sys.stderr)
        sys.exit(2)

    try:
        excludes = resolve_excludes(args.exclude, args.exclude_from)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)

    print(
        f"[info] 共 {len(abs_targets)} 个 FS 命中文件 + "
        f"{len(rel_patterns)} 个 target 模式 + "
        f"{len(excludes)} 个 exclude 规则：",
        file=sys.stderr,
    )
    for t in sorted(abs_targets):
        print(f"  [target abs] {t}", file=sys.stderr)
    for pat, _ in rel_patterns:
        print(f"  [target pat] {pat}", file=sys.stderr)
    for pat, _ in excludes:
        print(f"  [exclude]    {pat}", file=sys.stderr)

    start_trig = parse_trigger(args.start_at)
    stop_trig = parse_trigger(args.stop_at)

    if args.profile_hits is not None and start_trig is None:
        print(
            "[error] --profile-hits 需要配合 --start-at 使用（要靠 start trigger "
            "那行定位计数对象）",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.profile_hits is not None and args.profile_hits <= 0:
        print(
            f"[error] --profile-hits 必须为正整数，实际收到: {args.profile_hits}",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.max_duration is not None and args.max_duration <= 0:
        print(
            f"[error] --max-duration 必须为正数，实际收到: {args.max_duration}",
            file=sys.stderr,
        )
        sys.exit(2)

    finalize_done = [False]
    profiler_holder = [None]  # 闭包占位，让 finalize 能引用 profiler

    def finalize():
        if finalize_done[0]:
            return
        finalize_done[0] = True
        profiler = profiler_holder[0]
        if profiler is None:
            return
        try:
            profiler.stop()
        finally:
            print("\n" + "=" * 100, file=sys.stderr)
            # 记录汇总：在写入报告前打印，让用户能立即看到记录结果概况
            agg = profiler.aggregate()  # 已缓存，几乎零开销
            files_count = len({fn for fn, _ in agg.keys()})
            lines_count = len(agg)
            total_hits = sum(c for _, c in agg.values())
            total_in_target_ms = sum(t for t, _ in agg.values()) * 1000
            bucket_count = profiler.max_bucket + 1 if profiler._thread_state else 0
            rec_dur = profiler.recording_duration
            rec_dur_str = f"{rec_dur*1000:.2f} ms" if rec_dur is not None else "未记录"
            threads = profiler.list_threads()

            print("[TimedLineProfiler] 记录已结束，准备生成报告 ...", file=sys.stderr)
            print(f"  记录时长:     {rec_dur_str}", file=sys.stderr)
            print(f"  涉及文件数:   {files_count}", file=sys.stderr)
            print(f"  命中行数:     {lines_count} (去重)", file=sys.stderr)
            print(f"  总命中次数:   {total_hits}", file=sys.stderr)
            print(f"  目标内总耗时: {total_in_target_ms:.2f} ms", file=sys.stderr)
            print(
                f"  时间窗口:     {bucket_count} 个 × {args.bucket} s", file=sys.stderr
            )
            print(f"  涉及线程:     {len(threads)} 个", file=sys.stderr)
            for t in threads:
                tag = " [main]" if t["is_main"] else ""
                print(f"    - {t['name']:30s} (tid={t['tid']}){tag}", file=sys.stderr)
            print("=" * 100, file=sys.stderr)

            text = render_text(profiler, threshold_ms=args.threshold_ms)
            text_path = os.path.join(args.out, "report.txt")
            try:
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[ok] Text 报告已写入: {text_path}", file=sys.stderr)
            except OSError as e:
                print(f"[warn] 写入 {text_path} 失败: {e}", file=sys.stderr)
            html_path = os.path.join(args.out, "report.html")
            md_path = os.path.join(args.out, "report.md")
            render_html(
                profiler,
                html_path,
                top_k=args.top_k,
                top_ratio_pct=args.top_ratio,
                max_file_views=args.html_max_file_views,
                all_files=args.html_all_files,
            )
            render_markdown(
                profiler,
                md_path,
                top_k=args.top_k,
                top_ratio_pct=args.top_ratio,
                threshold_ms=args.threshold_ms,
            )

    profiler = TimedLineProfiler(
        target_files=abs_targets,
        bucket_seconds=args.bucket,
        start_trigger=start_trig,
        stop_trigger=stop_trig,
        cuda_sync=args.cuda_sync,
        target_patterns=rel_patterns,
        exclude_patterns=excludes,
        profile_hits=args.profile_hits,
        max_duration=args.max_duration,
        on_stop_callback=finalize,  # stop 触发时立即写报告，不必等程序结束
        main_thread_only_trigger=args.main_thread_only_trigger,
    )
    profiler_holder[0] = profiler

    atexit.register(finalize)

    sys.argv = [args.script] + (args.script_args or [])
    profiler.start()
    try:
        runpy.run_path(args.script, run_name="__main__")
    except KeyboardInterrupt:
        print("\n[info] 收到 KeyboardInterrupt，将生成报告 ...", file=sys.stderr)
    except SystemExit:
        # 用户脚本主动 sys.exit() 是正常路径，原样传递给上层
        raise
    except BaseException as e:
        # 训练脚本抛任何其它异常：吞掉而不重新抛，让 profiler 仍能输出已采集数据
        # （重抛会让 main() 异常退出，atexit 仍会跑，但 finally 里我们已经主动 finalize）
        import traceback

        print(f"\n[warn] 训练脚本异常退出: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "[info] profiler 将尝试输出已采集到的数据。完整异常 traceback：",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
    finally:
        # 立刻关 trace —— 防止异常处理 / cleanup 代码在 trace 激活下运行
        # 引发更多次级问题（典型场景：C++ 扩展销毁器中调用 Python 回调）
        try:
            profiler.stop()
        except Exception as e:
            print(f"[warn] profiler.stop() 失败: {e}", file=sys.stderr)
        # 主动调一次 finalize 落盘 —— 训练脚本若是 C++ abort()/SIGKILL
        # 终止进程会绕过 atexit，主动调用是唯一保险
        try:
            finalize()
        except Exception as e:
            print(f"[warn] finalize() 失败，已采集数据可能未写入: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
