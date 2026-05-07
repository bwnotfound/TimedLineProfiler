"""路径模式工具：glob → regex、target/exclude/trigger 解析。

这里是所有"输入字符串模式 → runtime 路径匹配 regex"的转换中心。三种用途共用同一套 glob 语法：
  - --target 目标文件
  - --exclude / --exclude-from 排除文件
  - --start-at / --stop-at 触发器

支持的 glob 元字符：
    *  -> [^/]*          单段内任意字符（不跨 /）
    ** -> .*             跨多层目录（独立出现时）
    **/ -> (?:.*/)?      0 或多个完整目录段（含尾随 /）
    ?  -> [^/]
    [...] -> 字符类（[!...] -> [^...]）

所有 regex 都不锚定到字符串开头，但要求匹配位置之前是字符串开头或 /，
即按"完整目录段"对齐，避免 'pandas/...' 误命中 'notpandas/...'。
"""

import glob as _glob
import os
import re
import sys
from typing import List, Optional, Set, Tuple


def glob_to_regex(pattern: str) -> "re.Pattern":
    """把路径 glob 转为 regex，开头自带 (?:^|/) 锚点，末尾 $ 锚定。

    使用 .search() 即可匹配。
    """
    pat = pattern.replace(os.sep, "/").strip("/")
    res: List[str] = ["(?:^|/)"]
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if i + 1 < n and pat[i + 1] == "*":
                if i + 2 < n and pat[i + 2] == "/":
                    res.append("(?:.*/)?")
                    i += 3
                    continue
                res.append(".*")
                i += 2
                continue
            res.append("[^/]*")
            i += 1
            continue
        elif c == "?":
            res.append("[^/]")
            i += 1
            continue
        elif c == "[":
            j = i + 1
            if j < n and pat[j] == "!":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                res.append(r"\[")
                i += 1
                continue
            cls = pat[i : j + 1]
            if cls.startswith("[!"):
                cls = "[^" + cls[2:]
            res.append(cls)
            i = j + 1
            continue
        elif c in r".\+(){}|^$":
            res.append("\\" + c)
            i += 1
            continue
        else:
            res.append(c)
            i += 1
    res.append("$")
    return re.compile("".join(res))


def resolve_targets(
    patterns: List[str],
) -> Tuple[Set[str], List[Tuple[str, "re.Pattern"]]]:
    """每个 --target 优先 FS glob 展开为绝对路径；FS 未命中时再 fallback 为
    相对路径 regex（用于 profile 第三方库）。

    避免 '**/*.py' 这类宽泛模式作为相对模式时误伤整个文件系统的所有 .py。
    """
    abs_files: Set[str] = set()
    regexes: List[Tuple[str, "re.Pattern"]] = []
    for pat in patterns:
        matched = _glob.glob(pat, recursive=True)
        py_matched = [m for m in matched if os.path.isfile(m) and m.endswith(".py")]
        if py_matched:
            for m in py_matched:
                abs_files.add(os.path.abspath(m))
        elif os.path.exists(pat) and pat.endswith(".py"):
            abs_files.add(os.path.abspath(pat))
        else:
            regexes.append((pat, glob_to_regex(pat)))
    return abs_files, regexes


def normalize_gitignore_line(line: str) -> Optional[str]:
    """gitignore 一行 -> glob 模式（None 表示跳过）。

    简化语义：
      - 空行、'#' 开头注释 -> None
      - '!' 否定 -> 哨兵 '__NEGATE__'，caller 应给 warn 后跳过
      - 末尾 '/'（目录）-> 转成 'pat/**'
      - 开头 '/'（根锚定）-> 去掉前导 '/'（_glob_to_regex 已在每个目录段开头匹配）
      - 其它 -> 原样
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("!"):
        return "__NEGATE__"
    is_dir = s.endswith("/")
    if is_dir:
        s = s.rstrip("/")
    if s.startswith("/"):
        s = s.lstrip("/")
    if not s:
        return None
    if is_dir:
        s = s + "/**"
    return s


def parse_exclude_file(path: str) -> List[Tuple[str, "re.Pattern"]]:
    """读 gitignore-like 文件，返回 [(原模式, regex), ...]。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--exclude-from 指向的文件不存在: {path}")
    out: List[Tuple[str, "re.Pattern"]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, raw in enumerate(f, 1):
            pat = normalize_gitignore_line(raw.rstrip("\n"))
            if pat is None:
                continue
            if pat == "__NEGATE__":
                print(
                    f"[warn] {path}:{ln_no} 使用了 '!' 否定语法，本工具不支持，"
                    f"已跳过该规则: {raw.rstrip()}",
                    file=sys.stderr,
                )
                continue
            out.append((pat, glob_to_regex(pat)))
    return out


def resolve_excludes(
    patterns: List[str],
    files: List[str],
) -> List[Tuple[str, "re.Pattern"]]:
    """汇总 --exclude 直接模式 + --exclude-from 文件中的所有规则。"""
    out: List[Tuple[str, "re.Pattern"]] = []
    for p in patterns:
        out.append((p, glob_to_regex(p)))
    for f in files:
        out.extend(parse_exclude_file(f))
    return out


def parse_trigger(spec: Optional[str]) -> Optional[Tuple["re.Pattern", int, int]]:
    """解析 'FILE:LINE' 或 'FILE:LINE:N'，N 缺省为 1。

    file_spec 支持具体文件 / glob / 相对路径模式。
    返回 (regex, line, n)。具体文件存在 -> 编译为该 abs path 的精确尾匹配 regex；
    否则 -> glob_to_regex。
    """
    if spec is None:
        return None
    # 优先尝试 FILE:LINE:N（最后两段都是纯数字）
    parts3 = spec.rsplit(":", 2)
    if len(parts3) == 3 and parts3[1].isdigit() and parts3[2].isdigit():
        fn, ln_s, n_s = parts3
        ln, n = int(ln_s), int(n_s)
    else:
        # 退而 FILE:LINE，N 默认 1
        parts2 = spec.rsplit(":", 1)
        if len(parts2) != 2 or not parts2[1].isdigit():
            raise ValueError(
                f"trigger 格式应为 FILE:LINE 或 FILE:LINE:N，实际收到: {spec}"
            )
        fn, ln, n = parts2[0], int(parts2[1]), 1
    if os.path.exists(fn):
        abs_norm = os.path.abspath(fn).replace(os.sep, "/")
        regex = re.compile(r"(?:^|/)" + re.escape(abs_norm.lstrip("/")) + "$")
    else:
        regex = glob_to_regex(fn)
    return (regex, ln, n)
