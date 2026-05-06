#!/usr/bin/env python3
"""
demo_train.py - 用于验证 line_time_profiler 的玩具脚本

包含：
    - 3 次 warmup（让你能用 --start-at 跳过）
    - 20 个 step 的"训练循环"（重计算 + 轻计算 + sleep）
    - 第 10 步开始故意让 heavy_op 变慢（模拟训练中的性能 drift），
      用 profiler 应该能在折线图里看到该行耗时上升

跑通验证：
    python line_time_profiler.py \
        --target demo_train.py \
        --bucket 0.3 \
        --start-at demo_train.py:50:1 \
        --stop-at  demo_train.py:50:21 \
        --out demo_report.html \
        -- demo_train.py
"""

import time

# 全局 flag，模拟某行随训练推进慢慢变慢
_slowdown_factor = 1


def heavy_op():
    """重计算：纯 Python 累加。"""
    s = 0
    n = 200_000 * _slowdown_factor
    for i in range(n):
        s += i * i
    return s


def light_op():
    """轻计算。"""
    return sum(range(2000))


def step(epoch):
    a = heavy_op()
    b = light_op()
    time.sleep(0.005)
    return a + b


def main():
    global _slowdown_factor
    print("warmup ...")
    for _ in range(3):
        step(-1)

    print("training ...")
    for epoch in range(20):                      # <-- 假设这是第 50 行附近
        if epoch == 10:
            _slowdown_factor = 4                  # 模拟 drift
        t0 = time.perf_counter()
        v = step(epoch)
        dt = (time.perf_counter() - t0) * 1000
        print(f"epoch {epoch:2d} done in {dt:7.2f} ms (val={v % 1000})")


if __name__ == '__main__':
    main()
