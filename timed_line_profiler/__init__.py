"""带时间窗口的逐行性能分析器。

使用示例：
    from timed_line_profiler import TimedLineProfiler

    profiler = TimedLineProfiler(target_files={"/abs/path/model.py"})
    profiler.start()
    ...  # run code
    profiler.stop()
"""

from .profiler import (
    TimedLineProfiler,
    main,
    render_html,
    render_text,
)

__version__ = "0.1.0"
__all__ = [
    "TimedLineProfiler",
    "render_html",
    "render_text",
    "main",
    "__version__",
]
