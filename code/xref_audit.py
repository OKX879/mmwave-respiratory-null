# -*- coding: utf-8 -*-
"""交叉引用体检：label/ref 完整性、图表编号顺序、cite 与作者名匹配
Cross-reference audit: label and ref completeness, figure and table numbering order, citation-author matching."""
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

for f in ['论文_多径零点_v5_TMTT.tex', '论文_多径零点_中文版.tex']:
    s = open(f, encoding='utf-8').read()
    body = s.split('\\begin{thebibliography}')[0]
    print('=' * 74)
    print(f)

    labels = set(re.findall(r'\\label\{([^}]+)\}', s))
    refs = re.findall(r'\\(?:eq)?ref\{([^}]+)\}', body)
    undef = [r for r in refs if r not in labels]
    unused = sorted(l for l in labels if l not in refs)
    print('  label %d | ref %d | 未定义引用: %s' % (len(labels), len(refs), undef or '无'))
    print('  定义但未引用的 label: %s' % (unused or '无'))

    # 图表编号顺序：按正文首次 \ref 出现的先后，检查 fig/table 是否递增
    order = []
    for m in re.finditer(r'\\ref\{(fig:[^}]+|tab:[^}]+)\}', body):
        k = m.group(1)
        if k not in order:
            order.append(k)
    figs = [k for k in order if k.startswith('fig:')]
    tabs = [k for k in order if k.startswith('tab:')]
    print('  Fig 首现顺序:', figs)
    print('  Tab 首现顺序:', tabs)

    # bib key 集合
    bibkeys = re.findall(r'\\bibitem\{([^}]+)\}', s)
    cites = []
    for m in re.finditer(r'\\cite\{([^}]+)\}', body):
        cites += [k.strip() for k in m.group(1).split(',')]
    print('  引用未在bib:', sorted(set(c for c in cites if c not in bibkeys)) or '无')
    print('  bib未被引:', sorted(set(k for k in bibkeys if k not in cites)) or '无')

    # 作者名与 cite 的邻接关系（英文版才有 "Xxx et al. \cite"）
    if 'v5_TMTT' in f:
        for m in re.finditer(r'([A-Z][A-Za-z\-]+)(?: \\emph\{et~al\.\})? \\cite\{([^}]+)\}', body):
            print('    姓名邻接 -> %-14s \\cite{%s}' % (m.group(1), m.group(2)))
