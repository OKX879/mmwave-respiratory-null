#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 论文_小波基选择_v0.1.md 转成 IEEEtran 双栏 LaTeX（XeLaTeX + xeCJK）。

只做机械转换，不改内容：
  # / ## / ###      → section / subsection / subsubsection
  | 表 |            → booktabs tabular（>5 列自动 table* + resizebox）
  ![](path)         → figure + includegraphics
  ``` 代码块 ```     → verbatim
  > 引用            → quote
  - / 1. 列表        → itemize / enumerate
  **粗** *斜* `码`   → textbf / emph / texttt
  【N】              → \\cite{rN}
  $...$             → 原样保留

用法：python md2tex.py   → 论文_小波基选择_IEEE.tex
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("论文_小波基选择_v0.1.md")
OUT = Path("论文_小波基选择_IEEE.tex")

SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def esc(t: str) -> str:
    """纯文本转义（不碰 $...$ 与已生成的命令）。"""
    return "".join(SPECIAL.get(c, c) for c in t)


def inline(t: str) -> str:
    """行内：先保护 math/code/引用，再转义，再还原并加格式。"""
    store: list[str] = []

    def keep(m: re.Match) -> str:
        store.append(m.group(0))
        return f"\x00{len(store) - 1}\x00"

    def keep_cite(m: re.Match) -> str:
        store.append(f"\\cite{{r{m.group(1)}}}")
        return f"\x00{len(store) - 1}\x00"

    def keep_tab(m: re.Match) -> str:
        store.append(f"\\ref{{tab:{m.group(1)}}}")
        return f"\x00{len(store) - 1}\x00"

    t = re.sub(r"\$[^$]+\$", keep, t)
    t = re.sub(r"`[^`]+`", keep, t)
    t = re.sub(r"【(\d+)】", keep_cite, t)
    t = re.sub(r"表\s*(\d+)", keep_tab, t)
    t = t.replace("\\|", "\x01PIPE\x01")     # 转义的竖线先藏起来，否则会被转义成 \textbackslash
    t = t.replace("\\>", ">")                 # md 里 \"\\>3.125\" 只是转义引用符
    t = esc(t)
    # bold 先于 italic
    t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", t)
    t = t.replace("\x01PIPE\x01", "|")

    def back(m: re.Match) -> str:
        s = store[int(m.group(1))]
        if s.startswith("`"):
            inner = s[1:-1]
            return r"\texttt{" + esc(inner) + "}"
        return s

    t = re.sub(r"\x00(\d+)\x00", back, t)
    t = t.replace("—", "---").replace("–", "--")
    t = t.replace("×", r"$\times$").replace("≥", r"$\ge$").replace("≤", r"$\le$")
    t = t.replace("±", r"$\pm$").replace("≈", r"$\approx$").replace("→", r"$\rightarrow$")
    return t


def parse_table(lines: list[str], caption: str = "") -> str:
    def split_row(ln: str) -> list[str]:
        """按 | 切分，但先保护转义竖线 \\|（否则单元格会被切开）。"""
        s = ln.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        s = s.replace("\\|", "\x01")
        return [c.replace("\x01", "\\|").strip() for c in s.split("|")]

    rows = [split_row(ln) for ln in lines]
    header, body = rows[0], rows[2:]
    ncol = len(header)
    # 文本型表格 → 用可自动换行的列；数值表 → 用 l
    textual = any(len(c) > 8 for r in [header] + body for c in r)
    # 列多或文字总量大 → 跨双栏
    wide = ncol > 5 or (textual and sum(len(c) for r in [header] + body for c in r) > 120)
    cap = ""
    if caption:
        mt = re.match(r"^(?:表|Table)\s*(\d+)[　\s]*(.*)$", caption)
        if mt:
            num, caption = int(mt.group(1)), mt.group(2)
            cap = ("\\caption{" + inline(caption) + "}\n"
                   + f"\\label{{tab:{num}}}\n")
        else:
            cap = "\\caption{" + inline(caption) + "}\n"
    if textual and not wide:
        out = [r"\begin{tabularx}{\columnwidth}{" + "X" * ncol + "}", r"\toprule",
               " & ".join(inline(c) for c in header) + r" \\", r"\midrule"]
        for r in body:
            r = (r + [""] * ncol)[:ncol]
            out.append(" & ".join(inline(c) for c in r) + r" \\")
        out += [r"\bottomrule", r"\end{tabularx}"]
        return ("\\begin{table}[!t]\n\\centering\n" + cap + "\\footnotesize\n"
                + "\n".join(out) + "\n\\end{table}")
    if textual and wide:
        # 列宽按内容长度分配（CJK 记 2 个宽度），避免中文在词中间折行
        def widx(cell: str) -> int:
            return sum(2 if ord(c) > 0x2E7F else 1 for c in cell)

        w = [max(widx(r[j]) for r in [header] + body) for j in range(ncol)]
        mean = sum(w) / ncol
        w = [max(x, 0.55 * mean) for x in w]
        tot = sum(w)
        widths = [0.96 * x / tot for x in w]
        spec = "".join(r">{\raggedright\arraybackslash}p{" + f"{x:.3f}\\textwidth" + "}"
                       for x in widths)
    else:
        spec = "l" * ncol
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(inline(c).replace(r"\_", r"\_\allowbreak{}") for c in header)
           + r" \\", r"\midrule"]
    for r in body:
        r = (r + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    tab = "\n".join(out)
    if wide:
        tab = (r"\resizebox{\textwidth}{!}{%" + "\n" + tab + "\n}")
        env = "table*"
    else:
        env = "table"
    return ("\\begin{" + env + "}[!t]\n\\centering\n" + cap + "\\footnotesize\n"
            + tab + "\n\\end{" + env + "}")


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    code: list[str] = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        # 代码块
        if s.startswith("```"):
            if not in_code:
                in_code, code = True, []
            else:
                in_code = False
                maxlen = max((len(x) for x in code), default=0)
                if maxlen > 45:            # 跨栏放，避免溢出单栏
                    out.append("\\begin{figure*}[!t]\n\\centering\n\\footnotesize\n"
                               "\\begin{verbatim}\n" + "\n".join(code)
                               + "\n\\end{verbatim}\n\\end{figure*}")
                else:
                    out.append("\\begin{verbatim}\n" + "\n".join(code)
                               + "\n\\end{verbatim}")
            i += 1
            continue
        if in_code:
            code.append(ln)
            i += 1
            continue

        # 图片
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            cap, path = m.group(1), m.group(2)
            out.append("\\begin{figure}[!t]\n\\centering\n"
                       f"\\includegraphics[width=\\columnwidth]{{{path}}}\n"
                       f"\\caption{{{inline(cap)}}}\n\\end{{figure}}")
            i += 1
            continue

        # 表格
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            # 紧邻上一行若是 "**表 …**"，取作 caption
            cap = ""
            k = i - 1
            while k >= 0 and lines[k].strip() == "":
                k -= 1
            if k >= 0:
                mc = re.match(r"^\*\*((?:表|Table).*)\*\*$", lines[k].strip())
                if mc:
                    cap = mc.group(1).strip()
                    while out and out[-1].strip() == "":
                        out.pop()
                    if out:
                        out.pop()
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.append(parse_table(tbl, cap))
            continue

        # 标题（去掉手写编号，交给 LaTeX 自动编号）
        if s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            txt = s[lvl:].strip()
            if lvl == 1:
                i += 1          # 题目单独处理
                continue
            txt = re.sub(r"^\d+(?:\.\d+)*\s*[、.]?\s*", "", txt)
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}.get(lvl, "paragraph")
            out.append(f"\\{cmd}{{{inline(txt)}}}")
            i += 1
            continue

        # 分隔线
        if re.match(r"^-{3,}$", s):
            i += 1
            continue

        # 引用块
        if s.startswith(">"):
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip().lstrip(">").strip())
                i += 1
            body = "\n\n".join(inline(x) for x in q if x)
            out.append("\\begin{quote}\\small\n" + body + "\n\\end{quote}")
            continue

        # 列表（支持按缩进嵌套）
        if re.match(r"^([-*]|\d+\.)\s+", s):
            entries: list[tuple[int, bool, str]] = []   # (indent, ordered, text)
            while i < len(lines):
                raw = lines[i]
                cur = raw.strip()
                ind = len(raw) - len(raw.lstrip())
                mm = re.match(r"^([-*]|\d+\.)\s+(.*)$", cur)
                if mm:
                    entries.append((ind, bool(re.match(r"^\d+\.", cur)), mm.group(2)))
                elif cur == "":
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines) and re.match(r"^([-*]|\d+\.)\s+", lines[j].strip()):
                        i = j
                        continue
                    break
                elif ind > 0 and entries:
                    entries[-1] = (entries[-1][0], entries[-1][1],
                                   entries[-1][2] + " " + cur.lstrip("> ").strip())
                else:
                    break
                i += 1

            def build(ent: list[tuple[int, bool, str]], lvl: int) -> tuple[str, int]:
                env = "enumerate" if ent[0][1] else "itemize"
                body: list[str] = []
                k = 0
                while k < len(ent):
                    ind, _, txt = ent[k]
                    if ind < lvl:
                        break
                    if ind > lvl and body:
                        sub, used = build(ent[k:], ind)
                        body[-1] += "\n" + sub
                        k += used
                        continue
                    body.append("\\item " + inline(txt))
                    k += 1
                return ("\\begin{" + env + "}\n" + "\n".join(body)
                        + "\n\\end{" + env + "}", k)

            block, _ = build(entries, entries[0][0])
            out.append(block)
            continue

        if s == "":
            out.append("")
            i += 1
            continue

        out.append(inline(ln))
        i += 1

    return "\n".join(out)


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    title = md.split("\n")[0].lstrip("# ").strip()
    full_md, _, refs_md = md.partition("## 参考文献")
    # 正文只取 "## 1 " 到 "## 附：稿件写作待办"（后者是内部清单，不进论文）
    body_md = full_md[full_md.index("## 1 "):]
    body_md = body_md.split("## 附：稿件写作待办")[0]

    body = convert(body_md)
    # 摘要（从完整 md 取，勿用截断后的 body_md）
    abst = ""
    m = re.search(r"## (?:摘要|Abstract)\n(.*?)\n---", full_md, re.S)
    if m:
        blocks = [b.strip() for b in m.group(1).split("\n\n") if b.strip()]
        kwkey = ("**关键词**", "**Index Terms**")
        abst = "\n\n".join(inline(b) for b in blocks if not b.startswith(kwkey))
        kw = next((b for b in blocks if b.startswith(kwkey)), "")
        kw = re.sub(r"^\*\*(?:关键词|Index Terms)\*\*", "", kw).strip()
    else:
        kw = ""

    # 参考文献：按编号排序，保证 \cite{rN} 的显示序号就是 N
    items_by_num: dict[int, str] = {}
    for ln in refs_md.split("\n"):
        mm = re.match(r"^\[(\d+)\]\s+(.*)$", ln.strip())
        if mm:
            items_by_num[int(mm.group(1))] = inline(mm.group(2))
    items = [f"\\bibitem{{r{n}}}\n{items_by_num[n]}" for n in sorted(items_by_num)]
    bib = "\n\n".join(items)

    tex = rf"""\documentclass[journal]{{IEEEtran}}
\usepackage{{xeCJK}}
\setCJKmainfont{{SimSun}}
\setCJKsansfont{{Microsoft YaHei}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\usepackage{{url}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{newunicodechar}}
\usepackage{{amssymb}}
\newunicodechar{{ρ}}{{\ensuremath{{\rho}}}}
\newunicodechar{{η}}{{\ensuremath{{\eta}}}}
\newunicodechar{{σ}}{{\ensuremath{{\sigma}}}}
\newunicodechar{{τ}}{{\ensuremath{{\tau}}}}
\newunicodechar{{ω}}{{\ensuremath{{\omega}}}}
\newunicodechar{{∈}}{{\ensuremath{{\in}}}}
\newunicodechar{{≠}}{{\ensuremath{{\neq}}}}
\newunicodechar{{≫}}{{\ensuremath{{\gg}}}}
\newunicodechar{{∝}}{{\ensuremath{{\propto}}}}
\newunicodechar{{↔}}{{\ensuremath{{\leftrightarrow}}}}
\newunicodechar{{⟺}}{{\ensuremath{{\Longleftrightarrow}}}}
\newunicodechar{{∩}}{{\ensuremath{{\cap}}}}
\newunicodechar{{≤}}{{\ensuremath{{\le}}}}
\newunicodechar{{≥}}{{\ensuremath{{\ge}}}}
\newunicodechar{{✓}}{{\ensuremath{{\checkmark}}}}
\newunicodechar{{✗}}{{\ensuremath{{\times}}}}
\newunicodechar{{★}}{{\ensuremath{{\star}}}}
\newunicodechar{{⚠}}{{\textbf{{!}}}}
\newunicodechar{{①}}{{(1)}}
\newunicodechar{{②}}{{(2)}}
\newunicodechar{{③}}{{(3)}}
\newunicodechar{{₁}}{{\textsubscript{{1}}}}
\newunicodechar{{₃}}{{\textsubscript{{3}}}}
\newunicodechar{{ₙ}}{{\textsubscript{{n}}}}
\graphicspath{{{{./}}}}
\hyphenpenalty=10000
\sloppy

\begin{{document}}

\title{{{inline(title)}}}
\author{{<作者待填>$^{{1}}$，<作者待填>$^{{2}}$\\[2pt]
$^{{1}}$<单位待填> \quad $^{{2}}$<单位待填>\\[2pt]
通讯作者：<待填>（email: <待填>）}}

\maketitle

\begin{{abstract}}
{abst}
\end{{abstract}}

\begin{{IEEEkeywords}}
{inline(kw)}
\end{{IEEEkeywords}}

{body}

\begin{{thebibliography}}{{99}}
{bib}
\end{{thebibliography}}

\end{{document}}
"""
    # 去掉转换时混进正文的摘要块（避免重复）
    tex = tex.replace("\\section*{摘要}", "")
    OUT.write_text(tex, encoding="utf-8")
    print(f"[ok] {OUT}  {len(tex)} bytes, bib items {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
