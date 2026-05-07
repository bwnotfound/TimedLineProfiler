"""HTML 报告：plotly 交互式可视化。

特性：
  - 顶部文件 dropdown：默认"全部文件"视图（所有选中行的并集），
    切换后热力图/折线/表格/subplot title 同步更新为该文件的视图
  - 自适应 log scale：动态范围 > 50 倍时启用，避免 outlier 压扁画面
  - 高度按全局视图行数自适应

实现：每个视图各自构造一组 (heatmap, lines, table) traces，
预先全部加入 figure，通过 dropdown 切换 visibility 显示。
subplot title 用 plotly 的 'annotations[i].text' dot-path 更新。
"""

import linecache
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from ..profiler import TimedLineProfiler
from ..selection import select_keys_per_file


def _build_subplot_titles(
    label: str, n_keys: int, n_lines: int, bucket_count: int, bs: float, log_tag: str
) -> Tuple[str, str, str]:
    """为某视图生成三个 subplot title。"""
    return (
        f"热力图: {n_keys} 行 × {bucket_count} 时间窗口（{bs}s/窗口）{log_tag}",
        f"折线图: Top-{n_lines} 行的耗时随时间变化{log_tag}",
        f"选中 {n_keys} 行汇总 — {label}",
    )


def _heatmap_colorbar(use_log: bool, max_in_view: float, epsilon: float):
    """为热力图构造 colorbar 配置。log 模式下 tick 仍标注真实 ms。"""
    if not use_log:
        return dict(title="ms")
    upper = max(max_in_view * 10, 0.01)
    ms_ticks = [
        t
        for t in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        if epsilon <= t <= upper
    ]
    return dict(
        title="ms (log)",
        tickvals=[math.log10(t + epsilon) for t in ms_ticks],
        ticktext=[(f"{t:g}" if t >= 1 else f"{t:.3g}") for t in ms_ticks],
    )


def render_html(
    profiler: TimedLineProfiler,
    out_path: str,
    top_k: int = 10,
    top_ratio_pct: float = 0.0,
    max_file_views: int = 8,
    all_files: bool = False,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print(
            "[warn] 未安装 plotly，跳过 HTML 报告。pip install plotly 可启用。",
            file=sys.stderr,
        )
        return

    agg = profiler.aggregate()
    if not agg:
        print("[warn] 没有数据，跳过 HTML。", file=sys.stderr)
        return

    bs = profiler.bucket_seconds
    bucket_count = profiler.max_bucket + 1
    bucket_labels = [f"{i*bs:.1f}-{(i+1)*bs:.1f}s" for i in range(bucket_count)]
    total_ms_global = sum(t for t, _ in agg.values()) * 1000

    # ---- 按文件分组（一次扫描，给后续单文件视图复用，避免重复 O(M) 扫描） ----
    by_file_keys: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for k, (t, _) in agg.items():
        by_file_keys[k[0]].append(k)
        file_total[k[0]] += t

    # ---- 决定要生成哪些视图 ----
    views: List[Tuple[str, List[Tuple[str, int]], List[List[float]], float]] = []

    # 全局视图（始终有）
    global_keys = select_keys_per_file(agg, top_k, top_ratio_pct)
    global_matrix = profiler.per_bucket_matrix(global_keys)
    views.append(
        (
            f"全部文件 ({total_ms_global:.0f} ms)",
            global_keys,
            global_matrix,
            total_ms_global,
        )
    )

    # 单文件视图：按文件耗时降序，受 max_file_views / all_files 限制
    files_sorted = sorted(by_file_keys.keys(), key=lambda f: -file_total[f])
    files_for_views = files_sorted if all_files else files_sorted[:max_file_views]
    omitted_count = len(files_sorted) - len(files_for_views)

    for fn in files_for_views:
        # 用预分组好的 keys 构造 sub_agg，避免再次扫整个 agg（旧代码 O(F*M)，新代码 O(M)）
        sub_agg = {k: agg[k] for k in by_file_keys[fn]}
        keys = select_keys_per_file(sub_agg, top_k, top_ratio_pct)
        matrix = profiler.per_bucket_matrix(keys)
        f_ms = file_total[fn] * 1000
        label = f"{os.path.basename(fn)} ({f_ms:.0f} ms)"
        views.append((label, keys, matrix, f_ms))

    if omitted_count > 0:
        print(
            f"[info] HTML dropdown 仅含 top-{len(files_for_views)} 文件视图，"
            f"其余 {omitted_count} 个文件未单独建视图（完整数据见 report.md）。"
            f"如需全部，加 --html-all-files。",
            file=sys.stderr,
        )

    # ---- log scale 决策（基于全局视图的动态范围）----
    all_pos = [v for row in global_matrix for v in row if v > 0]
    use_log = bool(all_pos) and (max(all_pos) / min(all_pos) > 50)
    epsilon = 1e-3
    log_tag = " [log scale]" if use_log else ""

    # ---- 创建 figure，subplot title 用全局视图的初始值 ----
    init_titles = _build_subplot_titles(
        views[0][0],
        len(views[0][1]),
        min(10, len(views[0][1])),
        bucket_count,
        bs,
        log_tag,
    )
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.4, 0.25, 0.35],
        subplot_titles=init_titles,
        specs=[[{"type": "heatmap"}], [{"type": "scatter"}], [{"type": "table"}]],
        vertical_spacing=0.06,
    )

    # ---- 为每个视图构造 traces，最后一次性 add_traces（plotly add_trace 单次 deepcopy
    # 开销较大，批量调用可省 ~10-20%；更主要是与限制视图数叠加） ----
    all_traces: List = []
    all_rows: List[int] = []
    all_cols: List[int] = []
    view_indices: List[List[int]] = []
    cur = 0

    for v_idx, (label, keys, matrix, denom_ms) in enumerate(views):
        idx_set: List[int] = []
        n_keys = len(keys)
        is_default = v_idx == 0
        row_labels = [f"{os.path.basename(fn)}:{ln}" for fn, ln in keys]
        code_snips = [linecache.getline(fn, ln).strip()[:80] for fn, ln in keys]
        # legend 单独使用更丰富的 label：行号 + 代码片段，便于看曲线时识别
        legend_labels = [
            f"{row_labels[i]} │ {code_snips[i][:40]}" for i in range(len(keys))
        ]

        # heatmap z（log 时取 log10）
        if use_log:
            z = [[math.log10(v + epsilon) for v in row] for row in matrix]
        else:
            z = matrix

        max_in_view = max(
            (v for row in matrix for v in row if v > 0),
            default=0.0,
        )
        colorbar = _heatmap_colorbar(use_log, max_in_view, epsilon)

        hover = [
            [
                f"行: {row_labels[i]}<br>窗口: {bucket_labels[j]}<br>"
                f"耗时: {matrix[i][j]:.3f} ms<br>代码: {code_snips[i]}"
                for j in range(bucket_count)
            ]
            for i in range(n_keys)
        ]

        all_traces.append(
            go.Heatmap(
                z=z,
                x=bucket_labels,
                y=row_labels,
                colorscale="Viridis",
                colorbar=colorbar,
                text=hover,
                hoverinfo="text",
                visible=is_default,
                showscale=True,
            )
        )
        all_rows.append(1)
        all_cols.append(1)
        idx_set.append(cur)
        cur += 1

        # 折线（top 10）
        n_lines = min(10, n_keys)
        for i in range(n_lines):
            all_traces.append(
                go.Scatter(
                    x=bucket_labels,
                    y=matrix[i],
                    mode="lines+markers",
                    name=legend_labels[i],
                    visible=is_default,
                    hovertemplate=f"{row_labels[i]}<br>%{{x}}: %{{y:.3f}} ms<extra></extra>",
                )
            )
            all_rows.append(2)
            all_cols.append(1)
            idx_set.append(cur)
            cur += 1

        # 表格
        pct_header = "占比" if v_idx == 0 else "文件内占比"
        table_rows = []
        for k in keys:
            fn, ln = k
            t, c = agg[k]
            ms = t * 1000
            avg = ms / c if c else 0.0
            pct = ms / denom_ms * 100 if denom_ms else 0.0
            code = linecache.getline(fn, ln).strip()
            table_rows.append(
                [
                    f"{os.path.basename(fn)}:{ln}",
                    f"{ms:.2f}",
                    str(c),
                    f"{avg:.4f}",
                    f"{pct:.1f}%",
                    code[:120],
                ]
            )

        all_traces.append(
            go.Table(
                header=dict(
                    values=[
                        "行",
                        "总耗时(ms)",
                        "调用次数",
                        "平均(ms)",
                        pct_header,
                        "代码",
                    ],
                    fill_color="#3a3a3a",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=(
                        list(zip(*table_rows)) if table_rows else [[] for _ in range(6)]
                    ),
                    align="left",
                    font=dict(size=11),
                    height=22,
                ),
                visible=is_default,
            )
        )
        all_rows.append(3)
        all_cols.append(1)
        idx_set.append(cur)
        cur += 1

        view_indices.append(idx_set)

    fig.add_traces(all_traces, rows=all_rows, cols=all_cols)
    total_traces = cur

    # ---- 每视图独立计算 figure 高度（dropdown 切换时同步更新） ----
    LEGEND_H = 110  # 底部水平 legend 预留空间

    def _calc_view_height(n_keys: int) -> int:
        heatmap_h = max(220, n_keys * 16 + 100)
        line_h = 320
        table_h = max(200, n_keys * 24 + 80)
        return heatmap_h + line_h + table_h + 250 + LEGEND_H

    # ---- dropdown buttons ----
    buttons = []
    for v_idx, (label, keys, _, _) in enumerate(views):
        visibility = [False] * total_traces
        for trace_i in view_indices[v_idx]:
            visibility[trace_i] = True
        n_keys = len(keys)
        n_lines = min(10, n_keys)
        new_titles = _build_subplot_titles(
            label,
            n_keys,
            n_lines,
            bucket_count,
            bs,
            log_tag,
        )
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "height": _calc_view_height(n_keys),
                        "annotations[0].text": new_titles[0],
                        "annotations[1].text": new_titles[1],
                        "annotations[2].text": new_titles[2],
                    },
                ],
            )
        )

    # ---- 初始高度按默认（第 0 个）视图行数定 ----
    total_h = _calc_view_height(len(global_keys))

    fig.update_layout(
        title=(
            f"TimedLineProfiler 报告（总耗时 {total_ms_global:.0f} ms · "
            f"{bucket_count} 个时间窗口 · {len(agg)} 个命中行 · "
            f"{len(files_sorted)} 个文件）"
        ),
        height=total_h,
        showlegend=True,
        # legend 水平排在整图下方，避开右侧的 colorbar；name 含代码片段便于识别
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dddddd",
            borderwidth=1,
            font=dict(size=10, family="monospace"),
            itemsizing="constant",
        ),
        margin=dict(b=LEGEND_H + 20),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                y=1.04,
                xanchor="left",
                yanchor="top",
                pad=dict(r=10, t=5),
                bgcolor="#f8f8f8",
                bordercolor="#cccccc",
                font=dict(size=12),
            )
        ],
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="文件视图：",
                x=0.0,
                y=1.075,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=12, color="#666"),
            )
        ],
    )
    fig.update_xaxes(title_text="时间窗口", row=1, col=1)
    fig.update_yaxes(title_text="文件:行号", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="时间窗口", row=2, col=1)
    fig.update_yaxes(title_text="耗时(ms)", row=2, col=1)
    if use_log:
        fig.update_yaxes(type="log", row=2, col=1)

    pio.write_html(fig, out_path, include_plotlyjs="cdn")
    print(f"[ok] HTML 报告已写入: {out_path}", file=sys.stderr)
