"""Parse the NPD (Sodir) July-2015 MySQL dump into parquet/csv tables."""
import re, sys, os
import pandas as pd

SRC = sys.argv[1]
OUT = sys.argv[2]
WANT = {'wellbore_exploration_all', 'discovery', 'discovery_reserves', 'field_reserves',
        'seis_acquisition', 'licence', 'wellbore_formation_top', 'field', 'seis_acquisition_coordinates_inc_turnarea'}
os.makedirs(OUT, exist_ok=True)
txt = open(SRC, encoding='utf-8', errors='replace').read()

cols = {}
for m in re.finditer(r'CREATE TABLE `(\w+)` \((.*?)\n\)', txt, re.S):
    t = m.group(1)
    cols[t] = re.findall(r'^\s+`(\w+)`', m.group(2), re.M)


def parse_values(s, i):
    """Parse '(..),(..);' starting at index i; yield tuples."""
    rows = []
    n = len(s)
    while i < n:
        c = s[i]
        if c == '(':
            row, i = [], i + 1
            while True:
                c = s[i]
                if c == "'":
                    i += 1
                    buf = []
                    while True:
                        c = s[i]
                        if c == '\\':
                            nx = s[i + 1]
                            buf.append({'n': '\n', 'r': '\r', 't': '\t', '0': '\0'}.get(nx, nx))
                            i += 2
                        elif c == "'":
                            if s[i + 1] == "'":
                                buf.append("'"); i += 2
                            else:
                                i += 1; break
                        else:
                            buf.append(c); i += 1
                    row.append(''.join(buf))
                else:
                    j = i
                    while s[j] not in ',)':
                        j += 1
                    tok = s[i:j].strip()
                    row.append(None if tok == 'NULL' else tok)
                    i = j
                c = s[i]
                if c == ',':
                    i += 1
                elif c == ')':
                    i += 1; break
            rows.append(row)
        elif c == ';':
            return rows, i + 1
        else:
            i += 1
    return rows, i


data = {t: [] for t in WANT}
for m in re.finditer(r'INSERT INTO `(\w+)` VALUES ', txt):
    t = m.group(1)
    if t not in WANT:
        continue
    rows, _ = parse_values(txt, m.end())
    data[t].extend(rows)

for t, rows in data.items():
    df = pd.DataFrame(rows, columns=cols[t])
    df.to_csv(f'{OUT}/{t}.csv', index=False)
    print(t, df.shape)
