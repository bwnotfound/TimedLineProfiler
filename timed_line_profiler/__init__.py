"""带时间窗口的逐行性能分析器。

公共 API：
    from timed_line_profiler import TimedLineProfiler, render_text, render_html, render_markdown

底层模块：
    .pattern    路径模式工具（glob_to_regex / resolve_targets / resolve_excludes / parse_trigger）
    .profiler   核心 TimedLineProfiler 类
    .selection  行选择策略（select_keys_per_file）
    .reports    三种报告渲染（reports.text / reports.md / reports.html）
    .cli        命令行入口
"""

from .cli import main
from .profiler import TimedLineProfiler
from .reports import render_html, render_markdown, render_text

__version__ = "0.1.1"
__all__ = [
    "TimedLineProfiler",
    "render_text",
    "render_html",
    "render_markdown",
    "main",
    "__version__",
]
