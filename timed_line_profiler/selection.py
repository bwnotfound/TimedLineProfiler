"""行选择策略：每个文件 max(top-k, ratio 触发数)，全局并集。"""

from collections import defaultdict
from typing import Dict, List, Tuple


def select_keys_per_file(
    agg: Dict[Tuple[str, int], List],
    top_k: int,
    top_ratio_pct: float,
) -> List[Tuple[str, int]]:
    """每个文件按耗时降序，选前 max(top_k, M) 行。

    M = 该文件中"占比 >= top_ratio_pct%"的行数。因占比单调下降，
    M 也是"占比 >= ratio 的行数"。

    返回所有选中 keys，按全局耗时降序排列。
    """
    by_file: Dict[str, List[Tuple[Tuple[str, int], float]]] = defaultdict(list)
    file_total: Dict[str, float] = defaultdict(float)
    for k, v in agg.items():
        t = v[0]
        by_file[k[0]].append((k, t))
        file_total[k[0]] += t

    selected: List[Tuple[Tuple[str, int], float]] = []
    for fn, items in by_file.items():
        items.sort(key=lambda x: -x[1])
        m = 0
        if top_ratio_pct > 0 and file_total[fn] > 0:
            thresh = file_total[fn] * top_ratio_pct / 100.0
            for i, (_, t) in enumerate(items, 1):
                if t >= thresh:
                    m = i
                else:
                    break
        n = max(top_k, m)
        selected.extend(items[:n])

    selected.sort(key=lambda x: -x[1])
    return [k for k, _ in selected]
