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
        '可重复指定。例如 "model/**/*.py"、'
        '"pandas/core/frame.py"、"torch/nn/modules/*.py"',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="排除文件的模式（与 --target 同 glob 语法）；可重复",
    )
    parser.add_argument(
        "--exclude-from",
        action="append",
        default=[],
        help="从 gitignore 风格文件读取排除规则；可重复。"
        "不支持 ! 否定语法（会跳过并 warn）",
    )
    parser.add_argument(
        "--bucket", type=float, default=5.0, help="时间窗口大小（秒），默认 5.0"
    )
    parser.add_argument(
        "--start-at",
        default=None,
        metavar="FILE:LINE:N",
        help="当 FILE 第 LINE 行被命中第 N 次时开始记录；"
        "FILE 同样支持 glob/相对路径",
    )
    parser.add_argument(
        "--stop-at",
        default=None,
        metavar="FILE:LINE:M",
        help="当 FILE 第 LINE 行被命中第 M 次时停止记录；"
        "FILE 同样支持 glob/相对路径",
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
    parser.add_argument("script", help="要执行的训练脚本")
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="传给训练脚本的参数"
    )

    args = parser.parse_args()

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

    profiler = TimedLineProfiler(
        target_files=abs_targets,
        bucket_seconds=args.bucket,
        start_trigger=start_trig,
        stop_trigger=stop_trig,
        cuda_sync=args.cuda_sync,
        target_patterns=rel_patterns,
        exclude_patterns=excludes,
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
            print(
                "[TimedLineProfiler] 训练结束/中断，开始生成报告 ...", file=sys.stderr
            )
            text = render_text(profiler, threshold_ms=args.threshold_ms)
            print(text)
            html_path = os.path.join(args.out, "report.html")
            md_path = os.path.join(args.out, "report.md")
            render_html(
                profiler, html_path, top_k=args.top_k, top_ratio_pct=args.top_ratio
            )
            render_markdown(
                profiler,
                md_path,
                top_k=args.top_k,
                top_ratio_pct=args.top_ratio,
                threshold_ms=args.threshold_ms,
            )

    atexit.register(finalize)

    sys.argv = [args.script] + (args.script_args or [])
    profiler.start()
    try:
        runpy.run_path(args.script, run_name="__main__")
    except KeyboardInterrupt:
        print("\n[info] 收到 KeyboardInterrupt，将生成报告 ...", file=sys.stderr)


if __name__ == "__main__":
    main()
