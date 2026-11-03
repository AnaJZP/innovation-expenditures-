"""Build the analysis datasets from INEGI public sources.

Inputs (datos_fuente/):
  innonce19_07.xlsx  Censos Economicos 2019, R&D expenditure of large units, 2016-2018
  innonce19_01.xlsx  Censos Economicos 2019, census indicators of large units (2018)
  pibt/              SCN quarterly GDP by activity (annual columns), current and 2018 prices
  ptf/               SCN KLEMS productivity accounts, 1990-2024
  mip/               Input-output matrix 2018, industry by industry, subsector level

Outputs (resultados/):
  panel_subsector.csv  86 census subsectors x 2016-2018, R&D + SCN GVA (commerce flagged)
  panel_klems.csv      64 KLEMS activities x 1990-2024 with R&D 2016-2018 merged
  io_weights.csv       64 x 64 domestic input-output flows aggregated to KLEMS activities
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datos_fuente"
OUT = ROOT / "resultados"
OUT.mkdir(exist_ok=True)
NA = ["ND", "NS", "NA", "-", "N/A"]

# ---------------------------------------------------------------- R&D (census)
x = pd.read_excel(SRC / "innonce19_07.xlsx", header=None)
nal = x[x[0].astype(str).str.startswith("00")]
sub = nal[nal[2].notna() & nal[3].isna()].copy()
sub["subsector"] = sub[2].str.extract(r"(\d{3})").astype(int)
cols = {8: "gastot_2016", 9: "gastot_2017", 10: "gastot_2018",
        11: "rd_2016", 12: "rd_2017", 13: "rd_2018",
        14: "sw_2016", 15: "sw_2017", 16: "sw_2018",
        17: "ip_2016", 18: "ip_2017", 19: "ip_2018",
        5: "units_total", 7: "units_with_rd"}
rd = sub[["subsector"] + list(cols)].rename(columns=cols)
rd = rd.apply(pd.to_numeric, errors="coerce").fillna(0)
rd_long = pd.wide_to_long(rd, ["gastot", "rd", "sw", "ip"], i="subsector", j="year", sep="_").reset_index()
rd_long[["gastot", "rd", "sw", "ip"]] /= 1000.0  # thousands -> millions of pesos

# census indicators of large units in 2018 (coverage of R&D universe)
c1n = pd.read_excel(SRC / "innonce19_01.xlsx", header=None)
c1n = c1n[c1n[0].astype(str).str.startswith("00")]
c1n = c1n[c1n[2].astype(str).str.contains("Subsector", na=False) & c1n[3].isna()]
cens = pd.DataFrame({
    "subsector": c1n[2].str.extract(r"(\d{3})")[0].astype(int).values,
    "census_po_large_2018": pd.to_numeric(c1n[8], errors="coerce").values,
    "census_va_large_2018": pd.to_numeric(c1n[17], errors="coerce").values,  # millions of pesos
})

# ------------------------------------------------------- SCN GVA by subsector
def read_pibt(fname, unit_label):
    d = pd.read_csv(SRC / "pibt/conjunto_de_datos" / fname, encoding="utf-8", na_values=NA)
    d = d[d.Descriptores.str.startswith(unit_label)]
    annual = [c for c in d.columns if c.endswith("|Anual") or re.match(r"^\d{4}\|Anual", c)]
    rows = []
    for _, r in d.iterrows():
        last = r.Descriptores.split("|")[-1]
        m = re.match(r"^([0-9\-]+)---", last)
        if not m:
            continue
        code = m.group(1)
        for c in annual:
            yr = int(c[:4])
            rows.append((code, yr, pd.to_numeric(r[c], errors="coerce")))
    out = pd.DataFrame(rows, columns=["code", "year", "value"]).drop_duplicates(["code", "year"])
    return out

gva_cur = read_pibt("conjunto_de_datos_pibtc_pibt_cor2026_t2.csv", "Millones de pesos a precios corrientes")
gva_cte = read_pibt("conjunto_de_datos_pibtc_pibt_cte2026_t2.csv", "Millones de pesos a precios de 2018")

# ---------------------------------------------------------------- KLEMS
KLEMS_FILES = {  # file number -> variable (identified from units and 2018 totals)
    152: "go_cur", 153: "go_real", 167: "va_cur", 168: "va_real", 169: "va_defl",
    170: "lab_comp", 172: "cap_comp", 174: "hours", 246: "persons",
    283: "gfcf_real", 286: "gfcf_ict_real", 291: "k_real", 292: "k_real_alt",
}

def read_klems(num):
    f = SRC / f"ptf/conjunto_de_datos/conjunto_de_datos_ptf152_295_ptf_{num}2024_r.csv"
    d = pd.read_csv(f, na_values=NA)
    d = d.iloc[: len(d) // 2]  # second block holds percentage changes
    d.columns = [re.sub(r"<.*>", "", c) for c in d.columns]
    d["code"] = d.Descriptores.str.split("|").str[-1].str.extract(r"^([0-9\-]+)---")[0]
    d = d.dropna(subset=["code"]).drop(columns="Descriptores").set_index("code")
    d = d.apply(pd.to_numeric, errors="coerce")
    return d.stack().rename(KLEMS_FILES[num]).rename_axis(["code", "year"])

kl = pd.concat([read_klems(n) for n in KLEMS_FILES], axis=1).reset_index()
kl["year"] = kl.year.astype(int)

# map the 86 census subsectors to KLEMS activities
def klems_code(s):
    groups = {"211-213-486": [211, 213, 486], "485-487": [485, 487], "491-492": [491, 492],
              "515-519": [515, 519], "517-518": [517, 518], "533-551": [533, 551],
              "561-562": [561, 562]}
    for g, members in groups.items():
        if s in members:
            return g
    if 431 <= s <= 437:
        return "430"
    if 461 <= s <= 469:
        return "460"
    return str(s)

rd_long["klems"] = rd_long.subsector.map(klems_code)
assert set(rd_long.klems) <= set(kl.code), set(rd_long.klems) - set(kl.code)

# ------------------------------------------------- subsector panel (86 x 3)
ps = rd_long.copy()
ps["code"] = ps.subsector.astype(str)
for name, src in [("gva_scn_cur", gva_cur), ("gva_scn_real", gva_cte)]:
    ps = ps.merge(src.rename(columns={"value": name}), on=["code", "year"], how="left")
ps["commerce"] = ps.subsector.between(431, 469)
# sectors 54, 55, 61 have a single subsector each; SCN reports them at sector level
for s, sec in [(541, "54"), (551, "55"), (611, "61")]:
    for name, src in [("gva_scn_cur", gva_cur), ("gva_scn_real", gva_cte)]:
        v = src[src.code == sec].set_index("year").value
        ps.loc[ps.subsector == s, name] = ps.loc[ps.subsector == s, "year"].map(v)
ps = ps.merge(cens, on="subsector", how="left")
ps.to_csv(OUT / "panel_subsector.csv", index=False)

# ------------------------------------------------ KLEMS activity panel
agg = rd_long.groupby(["klems", "year"])[["gastot", "rd", "sw", "ip"]].sum().reset_index()
names = (rd_long.drop_duplicates("subsector").groupby("klems").subsector.apply(lambda s: ",".join(map(str, sorted(s))))
         .rename("subsectors"))
pk = kl[kl.code.isin(agg.klems.unique())].rename(columns={"code": "klems"})
pk = pk.merge(agg, on=["klems", "year"], how="left").merge(names, on="klems")
pk["broad"] = pk.klems.str[0].map({"1": "primary", "2": "mining_util_constr", "3": "manufacturing"}).fillna("services")
pk.loc[pk.klems.str.startswith("21"), "broad"] = "mining_util_constr"
pk.to_csv(OUT / "panel_klems.csv", index=False)

# --------------------------------------------- input-output flows (domestic)
io = pd.read_csv(SRC / "mip/conjunto de datos/conjunto_de_datos_mip_d_pb_ixi_22018.csv", encoding="utf-8")
io["row"] = io.Descriptores.str.extract(r"^(\d{2,3})---")[0]
io = io.dropna(subset=["row"]).set_index("row")
dicols = {c: re.search(r"\|(\d{2,3})---", c).group(1) for c in io.columns
          if c.startswith("DI---") and re.search(r"\|(\d{2,3})---", c)}
Z = io[list(dicols)].rename(columns=dicols).apply(pd.to_numeric, errors="coerce").fillna(0)
Z = Z.groupby(level=0).sum().T.groupby(level=0).sum().T  # rows = supplier, cols = buyer

def io_to_klems(c):
    c = int(c) if c.isdigit() else c
    if c in (43, "43"):
        return "430"
    if c in (46, "46"):
        return "460"
    if c in (54, "54"):
        return "541"
    if c in (55, "55"):
        return "533-551"
    if c in (61, "61"):
        return "611"
    if isinstance(c, int):
        return klems_code(c)
    return None

rmap = {c: io_to_klems(c) for c in Z.index}
cmap = {c: io_to_klems(c) for c in Z.columns}
Zk = Z.groupby(rmap).sum().T.groupby(cmap).sum().T
keep = sorted(set(pk.klems))
Zk = Zk.reindex(index=keep, columns=keep).fillna(0)
Zk.to_csv(OUT / "io_weights.csv")
print("subsector panel:", ps.shape, "| KLEMS panel:", pk.shape, "| activities:", pk.klems.nunique(),
      "| IO matrix:", Zk.shape, "| IO missing rows:", [k for k in keep if Zk.loc[k].sum() == 0])
