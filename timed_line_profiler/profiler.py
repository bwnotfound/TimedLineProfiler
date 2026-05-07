"""核心 Profiler 类：基于 sys.settrace 的逐行耗时采集。

外围工具（路径模式、报告渲染、CLI）拆分到其它模块，本文件只负责：
  - 配置目标/排除/触发器
  - 注册 trace 钩子，按时间窗口聚合行耗时
  - 暴露 aggregate() / per_bucket_matrix() 给报告生成器消费
"""

import linecache
import os
import re
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


class TimedLineProfiler:
    def __init__(
        self,
        target_files: Set[str],
        bucket_seconds: float = 5.0,
        start_trigger: Optional[dict] = None,
        stop_trigger: Optional[dict] = None,
        cuda_sync: bool = False,
        target_patterns: Optional[List[Tuple[str, "re.Pattern"]]] = None,
        exclude_patterns: Optional[List[Tuple[str, "re.Pattern"]]] = None,
        profile_hits: Optional[int] = None,
        max_duration: Optional[float] = None,
        on_stop_callback=None,
        main_thread_only_trigger: bool = False,
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
        self._on_stop_callback = on_stop_callback
        self.main_thread_only_trigger = main_thread_only_trigger
        self.start_hit = 0
        self.stop_hit = 0
        self._profile_hits_count = 0
        self._start_consumed = False  # start trigger 是否已触发过（防止 stop 后被重启）
        self.recording = start_trigger is None

        # ---- per-thread 状态容器 ----
        # 每个线程的状态独立存储，避免线程间相互踩 last_line / bucket_data。
        # 结构：_thread_state[tid] = {
        #   'last_line': Optional[(fn, ln)],
        #   'last_time': Optional[float],
        #   'bucket_data':       defaultdict(int -> defaultdict((fn,ln) -> [t,c])),
        #   'func_bucket_data':  defaultdict(int -> defaultdict((fn,fname,fl) -> [t,c])),
        #   'meta': {
        #     'name': str,                # threading.current_thread().name
        #     'is_main': bool,
        #     'order': int,               # 进入 trace 的先后顺序（0 = 主线程，1+ = 子线程）
        #     'first_seen_perf': float,   # perf_counter 相对时间
        #     'first_seen_wall': float,   # wall clock 绝对时间戳
        #     'last_seen_perf': float,    # 持续更新
        #     'last_seen_wall': float,
        #   }
        # }
        self._thread_state: Dict[int, dict] = {}
        self._main_thread_id: Optional[int] = None

        self.start_wall: Optional[float] = None  # perf_counter 相对
        self.start_wall_absolute: Optional[float] = None  # time.time() 绝对（时间轴用）
        self.recording_duration: Optional[float] = None
        self.previous_trace = None
        self.enabled = False
        self.max_bucket = 0
        self._fname_cache: Dict[str, Optional[str]] = {}

        # 报告生成期的缓存（stop 后才会被填充，避免重复计算）
        # 缓存按 thread_filter 区分（None=合并；'main'=主；具体 tid=单线程）
        self._agg_caches: Dict = {}
        self._func_agg_caches: Dict = {}
        self._inverted_caches: Dict = {}

        # 函数级 frame 进入时刻：key 是 id(frame)，跨线程不冲突，dict 单步操作 GIL 原子
        self._frame_call_times: Dict[int, float] = {}

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
        trigger: dict,
        abs_fn: str,
        lineno: int,
        frame=None,
    ) -> bool:
        """检查 line 事件 (abs_fn, lineno) 是否命中 trigger。

        三种 kind：
        - 'line':  fn 匹配 file_re 且 lineno == trigger['line']
        - 'func':  fn 匹配 file_re 且 frame.f_code.co_name == func_name
                   且 lineno == frame.f_code.co_firstlineno + offset
        - 'regex': fn 匹配 file_re 且该行源码 search 命中 code_re（带缓存）
        """
        if not trigger["file_re"].search(abs_fn.replace(os.sep, "/")):
            return False
        kind = trigger["kind"]
        if kind == "line":
            return lineno == trigger["line"]
        if kind == "func":
            if frame is None:
                return False
            return (
                frame.f_code.co_name == trigger["func_name"]
                and lineno == frame.f_code.co_firstlineno + trigger["offset"]
            )
        if kind == "regex":
            cache = trigger["_cache"]
            cache_key = (abs_fn, lineno)
            if cache_key in cache:
                return cache[cache_key]
            line_text = linecache.getline(abs_fn, lineno)
            matched = bool(trigger["code_re"].search(line_text))
            cache[cache_key] = matched
            return matched
        return False

    def _do_stop_recording(self, now: float, reason: str):
        """统一的停止入口：触发任一 stop 条件时调用。

        关键动作（按顺序）：
          1. 标记 recording 结束并固化 recording_duration
          2. 立刻关全局 trace（sys.settrace + threading.settrace），让用户剩余代码原速运行
          3. 当前 frame 的 caller 应 return None 让 f_trace 也清掉
          4. 调用 on_stop_callback（通常是 cli 的 finalize）让报告立刻落盘
        """
        # threading 已在模块顶部导入
        self.recording = False
        self.recording_duration = now - self.start_wall
        # 清当前线程的 last_line（避免 stop 后该线程被再次触发 line 事件时累加）
        cur_state = self._thread_state.get(threading.get_ident())
        if cur_state is not None:
            cur_state["last_line"] = None
        sys.settrace(self.previous_trace)
        threading.settrace(None)
        self.enabled = False
        print(f"[info] 触发停止记录: {reason}", file=sys.stderr)
        print(
            "[info] 已关闭 sys.settrace；剩余用户代码将以原速运行（profile 数据已采集完毕）",
            file=sys.stderr,
        )
        if self._on_stop_callback is not None:
            try:
                self._on_stop_callback()
            except Exception as e:
                import traceback

                print(
                    f"[warn] on-stop 回调失败: {type(e).__name__}: {e}", file=sys.stderr
                )
                traceback.print_exc(file=sys.stderr)

    def _global_trace(self, frame, event, arg):
        if event != "call":
            return None
        if self._resolve(frame.f_code.co_filename) is None:
            return None
        # 第一次见到该线程时注册 thread_state（本路径在每个线程的 trace 启动后调一次）
        # threading 已在模块顶部导入
        tid = threading.get_ident()
        state = self._thread_state.get(tid)
        if state is None:
            state = self._get_or_init_thread_state(tid)
        # frame 进入时间（用于函数级累加；frame_id 跨线程不冲突）
        # generator 每次 next()/send() 也触发 call 事件，自然支持 yield 拆分计时。
        self._frame_call_times[id(frame)] = time.perf_counter()
        return self._local_trace

    def _local_trace(self, frame, event, arg):
        # threading 已在模块顶部导入
        tid = threading.get_ident()
        # state 必定已存在（_global_trace 中已建）；防御性 fallback
        state = self._thread_state.get(tid)
        if state is None:
            state = self._get_or_init_thread_state(tid)

        if event == "return":
            # 函数级 + 行级在 return 事件统一处理（per-thread）。
            # 函数级：从 _frame_call_times 取 call_time，累加到该线程的 func_bucket_data。
            #         elapsed 用 max(call_time, start_wall) 裁剪到 recording 区间内。
            # 行级：结算 last_line（return 行）真实耗时，避免跨函数边界把 caller 之后
            #       非 target 代码错误累加到 callee return 行上。
            # generator 的 yield 由 sys.settrace 拆为 return/call 配对，因此 yield 期间
            # 等待时间天然不计入函数耗时。
            fid = id(frame)
            call_time = self._frame_call_times.pop(fid, None)
            if not (self.recording and self.start_wall is not None):
                return None
            if self._cuda_sync_fn is not None:
                self._cuda_sync_fn()
            now = time.perf_counter()
            # 函数级累加（写入当前线程的 func_bucket_data）
            if call_time is not None:
                effective_start = (
                    call_time if call_time > self.start_wall else self.start_wall
                )
                if now > effective_start:
                    f_elapsed = now - effective_start
                    f_bucket = int(
                        (effective_start - self.start_wall) / self.bucket_seconds
                    )
                    if f_bucket < 0:
                        f_bucket = 0
                    if f_bucket > self.max_bucket:
                        self.max_bucket = f_bucket
                    fn_resolved = self._fname_cache.get(frame.f_code.co_filename)
                    if fn_resolved is not None:
                        func_key = (
                            fn_resolved,
                            frame.f_code.co_name,
                            frame.f_code.co_firstlineno,
                        )
                        slot = state["func_bucket_data"][f_bucket][func_key]
                        slot[0] += f_elapsed
                        slot[1] += 1
            # 行级累加（return 行真实耗时；写入当前线程的 bucket_data）
            last_line = state["last_line"]
            last_time = state["last_time"]
            if last_line is not None and last_time is not None:
                elapsed = now - last_time
                bucket = int((last_time - self.start_wall) / self.bucket_seconds)
                if bucket < 0:
                    bucket = 0
                if bucket > self.max_bucket:
                    self.max_bucket = bucket
                slot = state["bucket_data"][bucket][last_line]
                slot[0] += elapsed
                slot[1] += 1
            # 清空跨函数状态 + 更新该线程的 last_seen
            state["last_line"] = None
            state["last_time"] = None
            state["meta"]["last_seen_perf"] = now
            state["meta"]["last_seen_wall"] = time.time()
            return None  # frame 即将退出，不必再返回 trace 函数

        if event != "line":
            return self._local_trace

        fn = self._fname_cache.get(frame.f_code.co_filename)
        if fn is None:
            return self._local_trace
        lineno = frame.f_lineno
        key = (fn, lineno)

        # 是否仅主线程参与 trigger 检查
        is_main_thread = tid == self._main_thread_id
        trigger_eligible = (not self.main_thread_only_trigger) or is_main_thread

        if not self.recording:
            if (
                trigger_eligible
                and self.start_trigger is not None
                and not self._start_consumed
                and self._trigger_match(self.start_trigger, fn, lineno, frame)
            ):
                self.start_hit += 1
                if self.start_hit >= self.start_trigger["n"]:
                    if self._cuda_sync_fn is not None:
                        self._cuda_sync_fn()
                    self.recording = True
                    self._start_consumed = True
                    self.start_wall = time.perf_counter()
                    self.start_wall_absolute = time.time()
                    state["last_line"] = None
                    state["last_time"] = self.start_wall
                    print(
                        f"[info] 触发开始记录: {fn}:{lineno} 第 {self.start_hit} 次 "
                        f"(线程 {state['meta']['name']})",
                        file=sys.stderr,
                    )
            return self._local_trace

        if self._cuda_sync_fn is not None:
            self._cuda_sync_fn()

        now = time.perf_counter()

        # 行级累加：上一行 last_line（同线程内）的耗时
        last_line = state["last_line"]
        last_time = state["last_time"]
        if last_line is not None and last_time is not None:
            elapsed = now - last_time
            bucket = int((last_time - self.start_wall) / self.bucket_seconds)
            if bucket < 0:
                bucket = 0
            if bucket > self.max_bucket:
                self.max_bucket = bucket
            slot = state["bucket_data"][bucket][last_line]
            slot[0] += elapsed
            slot[1] += 1

        state["last_line"] = key
        state["last_time"] = now
        state["meta"]["last_seen_perf"] = now
        state["meta"]["last_seen_wall"] = time.time()

        # stop trigger（受 main_thread_only_trigger 控制）
        if (
            trigger_eligible
            and self.stop_trigger is not None
            and self._trigger_match(self.stop_trigger, fn, lineno, frame)
        ):
            self.stop_hit += 1
            if self.stop_hit >= self.stop_trigger["n"]:
                self._do_stop_recording(
                    now,
                    f"{fn}:{lineno} 第 {self.stop_hit} 次 "
                    f"(线程 {state['meta']['name']})",
                )
                return None

        # profile_hits：start trigger 那行被再命中 N 次
        if (
            trigger_eligible
            and self.profile_hits is not None
            and self.start_trigger is not None
            and self._trigger_match(self.start_trigger, fn, lineno, frame)
        ):
            self._profile_hits_count += 1
            if self._profile_hits_count >= self.profile_hits:
                self._do_stop_recording(
                    now,
                    f"--profile-hits 达到 {self._profile_hits_count} 次 "
                    f"(线程 {state['meta']['name']})",
                )
                return None

        # max_duration：开始后限时（与线程无关）
        if (
            self.max_duration is not None
            and self.start_wall is not None
            and now - self.start_wall >= self.max_duration
        ):
            self._do_stop_recording(
                now,
                f"--max-duration {self.max_duration}s 到时 "
                f"(实际 {now - self.start_wall:.3f}s)",
            )
            return None

        return self._local_trace

    # ---- 生命周期 ---------------------------------------------------------

    # ---- per-thread 状态管理 ----------------------------------------------

    def _get_or_init_thread_state(self, tid: int) -> dict:
        """获取或初始化指定线程的 state 容器。

        第一次见到一个 tid 时：
        - 创建空的 bucket_data / func_bucket_data
        - 初始化 last_line / last_time
        - 抓取 thread_meta（name、是否主线程、首次出现时刻）
        清除 _agg_caches 让下次 aggregate() 重新计算（因为有新数据源了）。
        """
        state = self._thread_state.get(tid)
        if state is not None:
            return state
        # threading 已在模块顶部导入
        try:
            tname = threading.current_thread().name
        except Exception:
            tname = f"Thread-{tid}"
        now_perf = time.perf_counter()
        now_wall = time.time()
        state = {
            "last_line": None,
            "last_time": None,
            "bucket_data": defaultdict(lambda: defaultdict(lambda: [0.0, 0])),
            "func_bucket_data": defaultdict(lambda: defaultdict(lambda: [0.0, 0])),
            "meta": {
                "name": tname,
                "is_main": (tid == self._main_thread_id),
                "order": len(self._thread_state),
                "first_seen_perf": now_perf,
                "first_seen_wall": now_wall,
                "last_seen_perf": now_perf,
                "last_seen_wall": now_wall,
            },
        }
        self._thread_state[tid] = state
        # 数据形状变了，缓存失效
        self._agg_caches.clear()
        self._func_agg_caches.clear()
        self._inverted_caches.clear()
        return state

    def start(self):
        # threading 已在模块顶部导入
        for f in self.target_files:
            self._fname_cache[f] = f
        # 必须在 settrace 之前定主线程 id；之后 sys.settrace 会让本线程也走 _global_trace
        self._main_thread_id = threading.get_ident()
        if self.recording:
            self.start_wall = time.perf_counter()
            self.start_wall_absolute = time.time()
            # 初始化主线程 state（这样 trace 第一次进来时 last_time 已有值）
            main_state = self._get_or_init_thread_state(self._main_thread_id)
            main_state["last_time"] = self.start_wall
        self.previous_trace = sys.gettrace()
        sys.settrace(self._global_trace)
        # 关键：让此后用 threading 创建的所有子线程都自动 trace。
        # 不影响已存在的线程，但 profiler 通常在 main 入口启动，那时只有主线程。
        threading.settrace(self._global_trace)
        self.enabled = True

    def stop(self):
        if not self.enabled:
            return
        # threading 已在模块顶部导入
        now = time.perf_counter()
        # 主线程的最后一行兜底累加（程序自然退出但 stop trigger 没触发的场景）
        # 子线程的 last_line 我们不主动累加 —— 子线程 join 后其 last_line 已是 None；
        # 还在跑的子线程也会随 trace 函数 fast-return 自然停止。
        if self.recording and self.start_wall is not None:
            main_tid = self._main_thread_id
            state = self._thread_state.get(main_tid) if main_tid is not None else None
            if (
                state is not None
                and state["last_line"] is not None
                and state["last_time"] is not None
            ):
                if self._cuda_sync_fn is not None:
                    self._cuda_sync_fn()
                    now = time.perf_counter()
                last_line = state["last_line"]
                last_time = state["last_time"]
                elapsed = now - last_time
                bucket = int((last_time - self.start_wall) / self.bucket_seconds)
                if bucket < 0:
                    bucket = 0
                if bucket > self.max_bucket:
                    self.max_bucket = bucket
                slot = state["bucket_data"][bucket][last_line]
                slot[0] += elapsed
                slot[1] += 1
                # 顺便更新主线程 last_seen
                state["meta"]["last_seen_perf"] = now
                state["meta"]["last_seen_wall"] = time.time()
            # recording_duration 兜底
            if self.recording_duration is None:
                self.recording_duration = now - self.start_wall
        # 关闭 trace：threading.settrace(None) 阻断新线程；sys.settrace(None) 关本线程；
        # 已存在的子线程 frame 上的 f_trace 仍生效但 enabled=False 时 fast-return。
        threading.settrace(None)
        sys.settrace(self.previous_trace)
        self.enabled = False

    # ---- 数据导出 ---------------------------------------------------------

    def _resolve_thread_filter(self, thread):
        """把 thread 参数标准化为要遍历的 tid 列表。

        - thread is None -> 所有线程（合并视图）
        - thread == 'main' -> 仅主线程
        - thread is int -> 指定 tid（不存在则空列表）
        """
        if thread is None:
            return list(self._thread_state.keys())
        if thread == "main":
            return (
                [self._main_thread_id]
                if self._main_thread_id in self._thread_state
                else []
            )
        if isinstance(thread, int):
            return [thread] if thread in self._thread_state else []
        raise ValueError(
            f"thread 参数必须是 None / 'main' / int(tid)，实际收到: {thread!r}"
        )

    def aggregate(self, thread=None) -> Dict[Tuple[str, int], List]:
        """聚合 (file, line) -> [total_time_s, count]。

        - thread=None: 合并所有线程（默认）
        - thread='main': 仅主线程
        - thread=<tid>: 指定线程 id

        结果按 thread filter 缓存。
        """
        cache_key = thread
        if cache_key in self._agg_caches:
            return self._agg_caches[cache_key]
        agg: Dict[Tuple[str, int], List] = defaultdict(lambda: [0.0, 0])
        for tid in self._resolve_thread_filter(thread):
            for bucket in self._thread_state[tid]["bucket_data"].values():
                for k, (t, c) in bucket.items():
                    agg[k][0] += t
                    agg[k][1] += c
        self._agg_caches[cache_key] = agg
        return agg

    def aggregate_funcs(self, thread=None) -> Dict[Tuple[str, str, int], List]:
        """聚合函数级 (file, func_name, first_lineno) -> [total_time_s, call_count]。

        - thread 参数语义同 aggregate()
        - call_count 是该函数被进入的次数；generator 每次 next()/send() 都算一次
        - total_time_s 是函数自身（含所有子调用）的执行时间，yield 之间等待不计入
        """
        cache_key = thread
        if cache_key in self._func_agg_caches:
            return self._func_agg_caches[cache_key]
        agg: Dict[Tuple[str, str, int], List] = defaultdict(lambda: [0.0, 0])
        for tid in self._resolve_thread_filter(thread):
            for bucket in self._thread_state[tid]["func_bucket_data"].values():
                for k, (t, c) in bucket.items():
                    agg[k][0] += t
                    agg[k][1] += c
        self._func_agg_caches[cache_key] = agg
        return agg

    def _build_inverted(
        self, thread=None
    ) -> Dict[Tuple[str, int], List[Tuple[int, float]]]:
        """构建倒排索引 (file, line) -> [(bucket_idx, ms), ...]。

        让 per_bucket_matrix(keys) 不必每次扫整个 bucket_data。
        thread 参数语义同 aggregate()。
        """
        cache_key = thread
        if cache_key in self._inverted_caches:
            return self._inverted_caches[cache_key]
        inv: Dict[Tuple[str, int], List[Tuple[int, float]]] = defaultdict(list)
        for tid in self._resolve_thread_filter(thread):
            for b, bucket in self._thread_state[tid]["bucket_data"].items():
                for k, (t, _) in bucket.items():
                    inv[k].append((b, t * 1000.0))
        self._inverted_caches[cache_key] = inv
        return inv

    def per_bucket_matrix(
        self, keys: List[Tuple[str, int]], thread=None
    ) -> List[List[float]]:
        """返回 [len(keys)][num_buckets] 的耗时矩阵，单位毫秒。"""
        n_b = self.max_bucket + 1 if self._thread_state else 1
        inv = self._build_inverted(thread)
        matrix = [[0.0] * n_b for _ in keys]
        for i, k in enumerate(keys):
            for b, ms in inv.get(k, ()):
                matrix[i][b] = ms
        return matrix

    def list_threads(self) -> List[dict]:
        """返回所有曾被 trace 的线程 meta 信息列表，按 order 升序。

        返回项格式::
            {'tid': int, 'name': str, 'is_main': bool, 'order': int,
             'first_seen_perf': float, 'first_seen_wall': float,
             'last_seen_perf': float,  'last_seen_wall': float}
        """
        out = []
        for tid, state in self._thread_state.items():
            m = state["meta"]
            out.append({"tid": tid, **m})
        out.sort(key=lambda x: x["order"])
        return out
