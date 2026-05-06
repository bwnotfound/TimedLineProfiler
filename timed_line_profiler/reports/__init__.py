"""三种格式的报告渲染：

- ``render_text``     LineProfiler 风格的文本汇总（每文件全行）
- ``render_markdown`` Markdown 报告（概览 + 各文件全行 + bucket 分布矩阵）
- ``render_html``     plotly 交互式 HTML（含文件 dropdown、自适应 log scale）
"""

from .html import render_html
from .md import render_markdown
from .text import render_text

__all__ = ["render_text", "render_markdown", "render_html"]
