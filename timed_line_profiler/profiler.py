"""核心 Profiler 类：基于 sys.settrace 的逐行耗时采集。

外围工具（路径模式、报告渲染、CLI）拆分到其它模块，本文件只负责：
  - 配置目标/排除/触发器
  - 注册 trace 钩子，按时间窗口聚合行耗时
  - 暴露 aggregate() / per_bucket_matrix() 给报告生成器消费
"""

import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


class TimedLineProfiler:
    def __init__(
        self,
        target_files: Set[str],
        bucket_seconds: float = 5.0,
        start_trigger: Optional[Tuple["re.Pattern", int, int]] = None,
        stop_trigger: Optional[Tuple["re.Pattern", int, int]] = None,
        cuda_sync: bool = False,
        target_patterns: Optional[List[Tuple[str, "re.Pattern"]]] = None,
        exclude_patterns: Optional[List[Tuple[str, "re.Pattern"]]] = None,
        profile_hits: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self.target_files: Set[str] = set(target_files)
        self.target_patterns: List[Tuple[str, "re.Pattern"]] = list(
            target_patterns or []
        )
        self.exclude_patterns: List[Tuple[str, "re.Pattern"]] = list(
            exclude_patterns or []
        )
        self.bucket_seconds = bucket_seconds
        self.start_trigger = start_trigger
        self.stop_trigger = stop_trigger
        self.profile_hits = profile_hits
        self.max_duration = max_duration
        self.start_hit = 0
        self.stop_hit = 0
        self._profile_hits_count = 0
        self._start_consumed = False  # start trigger 是否已触发过（防止 stop 后被重启）
        self.recording = start_trigger is None

        self.bucket_data: Dict[int, Dict[Tuple[str, int], List]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0])
        )

        self.start_wall: Optional[float] = None
        self.last_line: Optional[Tuple[str, int]] = None
        self.last_time: Optional[float] = None
        self.previous_trace = None
        self.enabled = False
        self.max_bucket = 0
        self._fname_cache: Dict[str, Optional[str]] = {}

        self._cuda_sync_fn = None
        if cuda_sync:
            try:
                import torch

                if not torch.cuda.is_available():
                    print(
                        "[warn] --cuda-sync 已开启但 CUDA 不可用，将跳过同步",
                        file=sys.stderr,
                    )
                else:
                    self._cuda_sync_fn = torch.cuda.synchronize
                    print(
                        "[info] --cuda-sync 已开启：每个目标行后将调用 "
                        "torch.cuda.synchronize()",
                        file=sys.stderr,
                    )
            except ImportError:
                print(
                    "[warn] --cuda-sync 已开启但未安装 torch，将跳过同步",
                    file=sys.stderr,
                )

        # 预过滤：把 exclude 命中的 target 文件从 target_files 里移除，
        # 避免 start() 预填 cache 时绕过 exclude 检查。
        if self.exclude_patterns and self.target_files:
            dropped = {
                f
                for f in self.target_files
                if self._is_excluded(f.replace(os.sep, "/"))
            }
            if dropped:
                self.target_files -= dropped
                for f in sorted(dropped):
                    print(f"[info] target 文件被 exclude 排除: {f}", file=sys.stderr)

    # ---- trace 钩子 -------------------------------------------------------

    def _is_excluded(self, abs_fn_norm: str) -> bool:
        for _, regex in self.exclude_patterns:
            if regex.search(abs_fn_norm):
                return True
        return False

    def _resolve(self, fn: str) -> Optional[str]:
        """raw filename -> abspath（命中目标）或 None。

        命中规则（按顺序）：
          1) 命中 exclude_patterns -> None
          2) abspath ∈ target_files -> 命中
          3) abspath 匹配 target_patterns 任一 regex -> 命中并加入 target_files
        """
        cached = self._fname_cache.get(fn, ...)
        if cached is not ...:
            return cached
        abs_fn = os.path.abspath(fn)
        norm = abs_fn.replace(os.sep, "/")

        if self.exclude_patterns and self._is_excluded(norm):
            self._fname_cache[fn] = None
            return None

        if abs_fn in self.target_files:
            result: Optional[str] = abs_fn
        elif self.target_patterns:
            result = None
            for _, regex in self.target_patterns:
                if regex.search(norm):
                    result = abs_fn
                    self.target_files.add(abs_fn)
                    break
        else:
            result = None

        self._fname_cache[fn] = result
        return result

    def _trigger_match(
        self,
        trigger: Tuple["re.Pattern", int, int],
        abs_fn: str,
        lineno: int,
    ) -> bool:
        regex, t_ln, _ = trigger
        if lineno != t_ln:
            return False
        return regex.search(abs_fn.replace(os.sep, "/")) is not None

    def _global_trace(self, frame, event, arg):
        if event != "call":
            return None
        if self._resolve(frame.f_code.co_filename) is None:
            return None
        return self._local_trace

    def _local_trace(self, frame, event, arg):
        if event != "line":
            return self._local_trace

        fn = self._fname_cache.get(frame.f_code.co_filename)
        if fn is None:
            return self._local_trace
        lineno = frame.f_lineno
        key = (fn, lineno)

        if not self.recording:
            if (
                self.start_trigger is not None
                and not self._start_consumed
                and self._trigger_match(self.start_trigger, fn, lineno)
            ):
                self.start_hit += 1
                if self.start_hit >= self.start_trigger[2]:
                    if self._cuda_sync_fn is not None:
                        self._cuda_sync_fn()
                    self.recording = True
                    self._start_consumed = True
                    self.start_wall = time.perf_counter()
                    self.last_line = None
                    self.last_time = self.start_wall
                    print(
                        f"[info] 触发开始记录: {fn}:{lineno} 第 {self.start_hit} 次",
                        file=sys.stderr,
                    )
            return self._local_trace

        if self._cuda_sync_fn is not None:
            self._cuda_sync_fn()

        now = time.perf_counter()

        if self.last_line is not None and self.last_time is not None:
            elapsed = now - self.last_time
            bucket = int((self.last_time - self.start_wall) / self.bucket_seconds)
            if bucket < 0:
                bucket = 0
            if bucket > self.max_bucket:
                self.max_bucket = bucket
            slot = self.bucket_data[bucket][self.last_line]
            slot[0] += elapsed
            slot[1] += 1

        self.last_line = key
        self.last_time = now

        if self.stop_trigger is not None and self._trigger_match(
            self.stop_trigger, fn, lineno
        ):
            self.stop_hit += 1
            if self.stop_hit >= self.stop_trigger[2]:
                self.recording = False
                self.last_line = None
                print(
                    f"[info] 触发停止记录: {fn}:{lineno} 第 {self.stop_hit} 次",
                    file=sys.stderr,
                )
                return self._local_trace

        # profile_hits：开始记录后再命中 start trigger 那行 N 次
        if (
            self.profile_hits is not None
            and self.start_trigger is not None
            and self._trigger_match(self.start_trigger, fn, lineno)
        ):
            self._profile_hits_count += 1
            if self._profile_hits_count >= self.profile_hits:
                self.recording = False
                self.last_line = None
                print(
                    f"[info] 触发停止记录: --profile-hits 达到 "
                    f"{self._profile_hits_count} 次",
                    file=sys.stderr,
                )
                return self._local_trace

        # max_duration：开始记录后限时
        if (
            self.max_duration is not None
            and self.start_wall is not None
            and now - self.start_wall >= self.max_duration
        ):
            self.recording = False
            self.last_line = None
            print(
                f"[info] 触发停止记录: --max-duration {self.max_duration}s 到时 "
                f"(实际 {now - self.start_wall:.3f}s)",
                file=sys.stderr,
            )
            return self._local_trace

        return self._local_trace

    # ---- 生命周期 ---------------------------------------------------------

    def start(self):
        for f in self.target_files:
            self._fname_cache[f] = f
        if self.recording:
            self.start_wall = time.perf_counter()
            self.last_time = self.start_wall
            self.last_line = None
        self.previous_trace = sys.gettrace()
        sys.settrace(self._global_trace)
        self.enabled = True

    def stop(self):
        if not self.enabled:
            return
        if (
            self.recording
            and self.last_line is not None
            and self.last_time is not None
            and self.start_wall is not None
        ):
            if self._cuda_sync_fn is not None:
                self._cuda_sync_fn()
            now = time.perf_counter()
            elapsed = now - self.last_time
            bucket = int((self.last_time - self.start_wall) / self.bucket_seconds)
            if bucket < 0:
                bucket = 0
            if bucket > self.max_bucket:
                self.max_bucket = bucket
            slot = self.bucket_data[bucket][self.last_line]
            slot[0] += elapsed
            slot[1] += 1
        sys.settrace(self.previous_trace)
        self.enabled = False

    # ---- 数据导出 ---------------------------------------------------------

    def aggregate(self) -> Dict[Tuple[str, int], List]:
        agg: Dict[Tuple[str, int], List] = defaultdict(lambda: [0.0, 0])
        for bucket in self.bucket_data.values():
            for k, (t, c) in bucket.items():
                agg[k][0] += t
                agg[k][1] += c
        return agg

    def per_bucket_matrix(self, keys: List[Tuple[str, int]]) -> List[List[float]]:
        """返回 [len(keys)][num_buckets] 的耗时矩阵，单位毫秒。"""
        n_b = self.max_bucket + 1 if self.bucket_data else 1
        matrix = [[0.0] * n_b for _ in keys]
        idx = {k: i for i, k in enumerate(keys)}
        for b, bucket in self.bucket_data.items():
            for k, (t, _) in bucket.items():
                if k in idx:
                    matrix[idx[k]][b] = t * 1000.0
        return matrix
