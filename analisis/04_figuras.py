"""Figures for the manuscript (print-ready: PNG 600 dpi plus vector SVG and PDF).

Palette validated with the data-viz checks (light mode, CVD and contrast):
plum #7b3f8c and jade #2a9d6f; marker shape is a second, colour-free encoding.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resultados"
FIG = RES / "figuras"
FIG.mkdir(exist_ok=True)

PLUM, JADE = "#7b3f8c", "#2a9d6f"
INK, MUTED, GRID, ZERO = "#1f1f1e", "#6b6a64", "#e8e6e1", "#9a988f"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "regular",
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, name):
    for ext, kw in [("png", {"dpi": 600}), ("svg", {}), ("pdf", {})]:
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    plt.close(fig)


def pointrange(ax, x, y, lo, hi, color, marker, label=None, horizontal=False, ms=6.0):
    if horizontal:
        ax.hlines(y, lo, hi, color=color, lw=1.6, alpha=0.9, zorder=2)
        ax.plot(x, y, marker, color=color, ms=ms, mec="white", mew=1.0, zorder=3, label=label)
    else:
        ax.vlines(x, lo, hi, color=color, lw=1.6, alpha=0.9, zorder=2)
        ax.plot(x, y, marker, color=color, ms=ms, mec="white", mew=1.0, zorder=3, label=label)


# ---------------------------------------------------------------- Figure 1
T = pd.read_excel(RES / "tablas_resultados.xlsx", sheet_name=None, index_col=0)
specs = [("B1_total_pooled_noctrl", "ln_gastot", "Pooled, no controls", "between"),
         ("B2_total_pooled_ctrl", "ln_gastot", "Pooled + capital and hours", "between"),
         ("A2c_pooled_total_sectorFE", "ln_gastot", "+ two-digit sector dummies", "between"),
         ("A4_twfe_total", "ln_gastot", "Two-way fixed effects", "within"),
         ("B5_total_fd", "d_ln_gastot", "First differences", "within"),
         ("B6_total_twfe_lag", "L_ln_gastot", "Two-way FE, spending lagged", "within")]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
ypos = np.arange(len(specs))[::-1]
for yy, (sh, var, lab, kind) in zip(ypos, specs):
    r = T[sh].loc[var]
    col, mk = (PLUM, "o") if kind == "between" else (JADE, "D")
    pointrange(ax, r.coef, yy, r.coef - 1.96 * r.se, r.coef + 1.96 * r.se, col, mk, horizontal=True,
               ms=6.5 if mk == "o" else 5.5)
    ax.text(r.coef + 1.96 * r.se + 0.015, yy, f"{r.coef:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
ax.axvline(0, color=ZERO, lw=0.9, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([s[2] for s in specs])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.06, 0.68)
ax.set_xlabel("Elasticity of real value added with respect to intangible spending (95% CI)")
ax.grid(axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.axhline(2.5, color=GRID, lw=0.9)
ax.text(0.665, 5.42, "Between industries", ha="right", va="center", fontsize=8.5, color=PLUM, style="italic")
ax.text(0.665, 2.12, "Within industries", ha="right", va="center", fontsize=8.5, color=JADE, style="italic")
ax.spines["left"].set_visible(False)
save(fig, "figure1_between_within")

# ---------------------------------------------------------------- Figure 2
lp = pd.read_csv(RES / "tabla_lp.csv")
panels = [("Total factor productivity", "TFP, winsorised intensity", "TFP, R&D only"),
          ("Real value added, given capital and hours", "GVA, winsorised intensity", "GVA, R&D only")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
for ax, (title, s_tot, s_rd) in zip(axes, panels):
    for spec, col, mk, off, lab in [(s_tot, PLUM, "o", -0.13, "All intangible spending"),
                                    (s_rd, JADE, "D", 0.13, "R&D only")]:
        d = lp[lp.spec == spec].sort_values("year")
        ax.plot(d.year + off, d.effect_1sd, color=col, lw=0.8, alpha=0.45, zorder=1)
        pointrange(ax, d.year + off, d.effect_1sd, d.ci_1sd_lo, d.ci_1sd_hi, col, mk, label=lab,
                   ms=5.8 if mk == "o" else 4.8)
    ax.axhline(0, color=ZERO, lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.axvspan(2019.5, 2020.5, color="#f1efea", zorder=-1, lw=0)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
    ax.set_xticks(range(2019, 2025))
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
axes[0].text(2020, 0.185, "pandemic", ha="center", fontsize=7.5, color=MUTED, style="italic")
axes[0].set_ylabel("Cumulative log change since 2018\nper 1 s.d. higher intensity")
h = [Line2D([], [], color=PLUM, marker="o", ls="-", lw=1.4, ms=5.5, mec="white"),
     Line2D([], [], color=JADE, marker="D", ls="-", lw=1.4, ms=4.6, mec="white")]
fig.legend(h, ["All intangible spending", "R&D only"], loc="lower center", ncol=2, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "figure2_local_projections")

# ---------------------------------------------------------------- Figure 3
SHORT = {"331": "Basic metals", "332": "Fabricated metal products", "523": "Securities and investment",
         "488": "Transport support services", "521": "Central banking", "493": "Warehousing",
         "335": "Electrical equipment", "336": "Transport equipment", "333": "Machinery and equipment",
         "491-492": "Postal and courier services", "533-551": "IP licensing; corporate management",
         "236": "Building construction", "325": "Chemicals", "522": "Credit intermediation",
         "811": "Repair and maintenance"}
emb = pd.read_csv(RES / "tabla_multiplicadores_io.csv", dtype={"klems": str})
e = emb[emb.spending == "rd"].sort_values("total", ascending=False).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.3, 4.3))
y = np.arange(len(e))
ax.hlines(y, e.direct * 1000, e.total * 1000, color="#d9d5ce", lw=2.4, zorder=1)
ax.plot(e.direct * 1000, y, "D", color=JADE, ms=5.2, mec="white", mew=0.9, zorder=3, label="Own R&D (direct)")
ax.plot(e.total * 1000, y, "o", color=PLUM, ms=6.2, mec="white", mew=0.9, zorder=3,
        label="Own + R&D embodied in domestic inputs (total)")
for yy, r in zip(y, e.itertuples()):
    ax.text(r.total * 1000 * 1.12, yy, f"×{r.multiplier:.1f}", va="center", fontsize=8, color=MUTED)
ax.set_xscale("log")
ax.set_xticks([0.02, 0.1, 0.5, 1, 2, 5, 15])
ax.set_xticklabels(["0.02", "0.1", "0.5", "1", "2", "5", "15"])
ax.minorticks_off()
ax.set_xlim(0.012, 30)
ax.set_yticks(y)
ax.set_yticklabels([f"{SHORT.get(k, k)} ({k})" for k in e.klems])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Pesos of R&D per 1,000 pesos of final demand, 2018 (log scale)")
ax.grid(axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.42, -0.30), ncol=1, fontsize=8.5)
save(fig, "figure3_io_rd_content")
print("figures saved to", FIG)
