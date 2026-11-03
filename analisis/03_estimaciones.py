"""Main estimations for the revised manuscript.

Design
  A. Static production-function panel, 64 KLEMS activities x 2016-2018, two-way FE,
     activity-clustered SEs with wild cluster bootstrap p-values.
  B. Long-difference local projections: cumulative growth 2018 -> 2018+h (h = 1..6) of
     real GVA and TFP on 2016-2018 intangible intensity (R/Y). The coefficient is a
     gross rate of return: pesos of additional value added per peso of spending.
  C. Input-output spillovers: R&D embodied in domestic purchases (Terleckyj weights).
  D. Heterogeneity by technology group and multiple-testing check of the
     subsector-interaction approach used in the first version.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados"
RNG = np.random.default_rng(20261030)
B = 1999
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ------------------------------------------------------------------ helpers
def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def crv1(X, u, g):
    """Cluster-robust (CR1) covariance."""
    n, k = X.shape
    G = len(np.unique(g))
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for c in np.unique(g):
        s = X[g == c].T @ u[g == c]
        meat += np.outer(s, s)
    adj = G / (G - 1) * (n - 1) / (n - k)
    return adj * XtXi @ meat @ XtXi


def hc3(X, u):
    XtXi = np.linalg.pinv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    w = (u / (1 - h)) ** 2
    return XtXi @ (X.T * w) @ X @ XtXi


def wild_p(y, X, j, g=None, B=B):
    """Wild (cluster) bootstrap p-value, null imposed (WCR / WR), Rademacher weights."""
    beta, u = ols(y, X)
    V = crv1(X, u, g) if g is not None else hc3(X, u)
    t0 = beta[j] / np.sqrt(V[j, j])
    Xr = np.delete(X, j, axis=1)
    br, ur = ols(y, Xr)
    fit_r = Xr @ br
    groups = np.unique(g) if g is not None else None
    ts = np.empty(B)
    for b in range(B):
        if g is not None:
            v = RNG.choice([-1.0, 1.0], size=len(groups))
            vv = v[np.searchsorted(groups, g)]
        else:
            vv = RNG.choice([-1.0, 1.0], size=len(y))
        ys = fit_r + ur * vv
        bs, us = ols(ys, X)
        Vs = crv1(X, us, g) if g is not None else hc3(X, us)
        ts[b] = bs[j] / np.sqrt(Vs[j, j])
    return float(np.mean(np.abs(ts) >= abs(t0)))


def fit(df, y, xs, g=None, se="cluster", boot=(), add_const=True):
    d = df.dropna(subset=[y] + xs)
    X = d[xs].to_numpy(float)
    if add_const:
        X = np.column_stack([np.ones(len(d)), X])
        names = ["const"] + xs
    else:
        names = xs
    yv = d[y].to_numpy(float)
    gv = d[g].to_numpy() if g is not None else None
    beta, u = ols(yv, X)
    V = crv1(X, u, gv) if se == "cluster" else hc3(X, u)
    sev = np.sqrt(np.diag(V))
    dof = (len(np.unique(gv)) - 1) if se == "cluster" else (len(yv) - X.shape[1])
    p = 2 * stats.t.sf(np.abs(beta / sev), dof)
    r2 = 1 - (u @ u) / (((yv - yv.mean()) ** 2).sum())
    res = pd.DataFrame({"coef": beta, "se": sev, "p": p}, index=names)
    res["p_boot"] = np.nan
    for v in boot:
        j = names.index(v)
        res.loc[v, "p_boot"] = wild_p(yv, X, j, gv if se == "cluster" else None)
    res.attrs.update(n=len(yv), r2=r2, clusters=len(np.unique(gv)) if gv is not None else None)
    return res


def twoway_demean(df, cols, i="klems", t="year"):
    d = df.copy()
    for c in cols:
        d[c] = d[c] - d.groupby(i)[c].transform("mean") - d.groupby(t)[c].transform("mean") + d[c].mean()
    return d


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


TABLES = {}


def keep(name, res, note=""):
    r = res.copy()
    r["n"] = res.attrs.get("n")
    r["r2"] = res.attrs.get("r2")
    r["clusters"] = res.attrs.get("clusters")
    r["note"] = note
    TABLES[name] = r
    log(f"\n== {name} {note}  N={res.attrs.get('n')} R2={res.attrs.get('r2'):.3f}")
    log(res.round(4).to_string())


# ------------------------------------------------------------------ data
pk = pd.read_csv(OUT / "panel_klems.csv", dtype={"klems": str})
pk = pk.sort_values(["klems", "year"])
pk["sector2"] = pk.klems.str[:2]
pk.loc[pk.klems.isin(["430", "460"]), "sector2"] = pk.klems.str[:2]

# labour share and TFP (value-added based, Tornqvist)
pk["sL"] = (pk.lab_comp / (pk.lab_comp + pk.cap_comp)).clip(0.05, 0.95)
pk = pk.sort_values(["klems", "year"])
g = pk.groupby("klems")
sbar = 0.5 * (pk.sL + g.sL.shift())
dtfp = (np.log(pk.va_real) - np.log(g.va_real.shift())
        - sbar * (np.log(pk.hours) - np.log(g.hours.shift()))
        - (1 - sbar) * (np.log(pk.k_real) - np.log(g.k_real.shift())))
pk["dtfp"] = dtfp
pk["ln_tfp"] = pk.groupby("klems").dtfp.transform(lambda s: s.fillna(0).cumsum())

# real intangible spending (deflated with the activity's value-added deflator)
for v in ["gastot", "rd", "sw", "ip"]:
    pk[v + "_real"] = pk[v] / (pk.va_defl / 100)

w = pk[pk.year.between(2016, 2018)].copy()
for v in ["gastot", "rd", "sw", "ip"]:
    w["ln_" + v] = np.arcsinh(w[v + "_real"])  # asinh keeps zeros; ~ln(2x) for large x
w["ln_va"] = np.log(w.va_real)
w["ln_k"] = np.log(w.k_real)
w["ln_h"] = np.log(w.hours)
w["ln_l"] = np.log(w.persons)

log("Zeros (activity-years, 2016-2018):", {v: int((w[v] == 0).sum()) for v in ["rd", "sw", "ip"]},
    "of", len(w))

# ------------------------------------------------------------ descriptives
desc_vars = {"va_real": "GVA (millions 2018 MXN)", "gastot_real": "Intangible spending, total",
             "rd_real": "R&D for innovation", "sw_real": "Software and databases",
             "ip_real": "Patents, trademarks, tech. packages", "k_real": "Capital stock (millions 2018 MXN)",
             "hours": "Hours worked (thousands)", "persons": "Persons employed"}
desc = w[list(desc_vars)].describe().T[["count", "mean", "50%", "std", "min", "max"]]
desc.index = [desc_vars[i] for i in desc.index]
desc.to_csv(OUT / "tabla_descriptivos.csv")
log("\nDescriptive statistics\n", desc.round(1).to_string())
corr = w[["ln_va", "ln_gastot", "ln_rd", "ln_sw", "ln_ip", "ln_k", "ln_h"]].corr()
corr.to_csv(OUT / "tabla_correlaciones.csv")
log("\nCorrelations\n", corr.round(2).to_string())

# ------------------------------------------------------------ A. static panel
X3 = ["ln_rd", "ln_sw", "ln_ip"]
keep("A1_pooled_noctrl", fit(w, "ln_va", X3, g="klems"), "pooled OLS, no controls")
keep("A2_pooled_ctrl", fit(w.assign(**{f"y{t}": (w.year == t).astype(float) for t in (2017, 2018)}),
                           "ln_va", X3 + ["ln_k", "ln_h", "y2017", "y2018"], g="klems"),
     "pooled OLS + capital + hours + year dummies")
sec_d = pd.get_dummies(w.sector2, prefix="s", drop_first=True, dtype=float)
w2 = pd.concat([w, sec_d], axis=1).assign(**{f"y{t}": (w.year == t).astype(float) for t in (2017, 2018)})
keep("A2b_pooled_ctrl_sectorFE", fit(w2, "ln_va", X3 + ["ln_k", "ln_h", "y2017", "y2018"] + list(sec_d.columns),
                                     g="klems", boot=X3), "pooled OLS + K + H + year + 2-digit sector dummies")
w2["ln_gastot"] = w.ln_gastot
keep("A2c_pooled_total_sectorFE", fit(w2, "ln_va", ["ln_gastot", "ln_k", "ln_h", "y2017", "y2018"] + list(sec_d.columns),
                                      g="klems", boot=["ln_gastot"]), "pooled OLS, total intangibles + K + H + year + sector dummies")
keep("A2_pooled_ctrl_boot", fit(w.assign(**{f"y{t}": (w.year == t).astype(float) for t in (2017, 2018)}),
                           "ln_va", X3 + ["ln_k", "ln_h", "y2017", "y2018"], g="klems", boot=X3),
     "pooled OLS + capital + hours + year dummies (bootstrap)")
wd = twoway_demean(w, ["ln_va"] + X3 + ["ln_gastot", "ln_k", "ln_h"])
keep("A3_twfe", fit(wd, "ln_va", X3 + ["ln_k", "ln_h"], g="klems", boot=X3, add_const=False),
     "two-way FE (activity + year)")
keep("A4_twfe_total", fit(wd, "ln_va", ["ln_gastot", "ln_k", "ln_h"], g="klems", boot=["ln_gastot"],
                          add_const=False), "two-way FE, total intangible spending")
# first differences with year effect
fd = w.sort_values(["klems", "year"]).copy()
for c in ["ln_va", "ln_k", "ln_h"] + X3:
    fd["d_" + c] = fd.groupby("klems")[c].diff()
fd["y2018"] = (fd.year == 2018).astype(float)
keep("A5_fd", fit(fd, "d_ln_va", [f"d_{c}" for c in X3] + ["d_ln_k", "d_ln_h", "y2018"], g="klems",
                  boot=[f"d_{c}" for c in X3]), "first differences")
# lagged spending (2016-2017 -> 2017-2018), within estimator
lag = w.sort_values(["klems", "year"]).copy()
for c in X3:
    lag["L_" + c] = lag.groupby("klems")[c].shift()
lag = lag.dropna(subset=["L_ln_rd"])
lagd = twoway_demean(lag, ["ln_va", "ln_k", "ln_h"] + [f"L_{c}" for c in X3])
keep("A6_twfe_lag", fit(lagd, "ln_va", [f"L_{c}" for c in X3] + ["ln_k", "ln_h"], g="klems",
                        boot=[f"L_{c}" for c in X3], add_const=False), "two-way FE, spending lagged one year")

# Panel B: total intangible spending in every specification
yd = {f"y{t}": (w.year == t).astype(float) for t in (2017, 2018)}
keep("B1_total_pooled_noctrl", fit(w, "ln_va", ["ln_gastot"], g="klems", boot=["ln_gastot"]), "total, pooled no controls")
keep("B2_total_pooled_ctrl", fit(w.assign(**yd), "ln_va", ["ln_gastot", "ln_k", "ln_h", "y2017", "y2018"], g="klems",
                                 boot=["ln_gastot"]), "total, pooled + K + H")
fd["ln_gastot"] = w.sort_values(["klems", "year"]).ln_gastot
fd["d_ln_gastot"] = fd.groupby("klems").ln_gastot.diff()
keep("B5_total_fd", fit(fd, "d_ln_va", ["d_ln_gastot", "d_ln_k", "d_ln_h", "y2018"], g="klems", boot=["d_ln_gastot"]),
     "total, first differences")
lag["L_ln_gastot"] = w.sort_values(["klems", "year"]).groupby("klems").ln_gastot.shift().loc[lag.index]
lagd2 = twoway_demean(lag, ["ln_va", "ln_k", "ln_h", "L_ln_gastot"])
keep("B6_total_twfe_lag", fit(lagd2, "ln_va", ["L_ln_gastot", "ln_k", "ln_h"], g="klems", boot=["L_ln_gastot"],
                              add_const=False), "total, two-way FE, lagged")

# Robustness of the production-function estimates (total intangible spending)
agg_defl = pk[pk.year.between(2016, 2018)].groupby("year").apply(lambda d: d.va_cur.sum() / d.va_real.sum() * 100)
w["ln_gastot_gdpdefl"] = np.arcsinh(w.gastot / (w.year.map(agg_defl) / 100))
w["ln_k_alt"] = np.log(w.k_real_alt)
sec_all = pd.get_dummies(w.sector2, prefix="s", drop_first=True, dtype=float)
wy = pd.concat([w, sec_all], axis=1).assign(**{f"y{t}": (w.year == t).astype(float) for t in (2017, 2018)})
SEC = list(sec_all.columns)
rob_pf = []


def pf_pair(label, d, x="ln_gastot", k="ln_k", lab="ln_h", g="klems"):
    d = d.copy()
    sec = [c for c in SEC if c in d.columns and d[c].std() > 0]
    rb = fit(d, "ln_va", [x, k, lab, "y2017", "y2018"] + sec, g=g, boot=[x])
    dd = twoway_demean(d, ["ln_va", x, k, lab])
    rw = fit(dd, "ln_va", [x, k, lab], g=g, boot=[x], add_const=False)
    rob_pf.append({"spec": label, "between": rb.loc[x, "coef"], "between_se": rb.loc[x, "se"],
                   "between_p": rb.loc[x, "p"], "between_pboot": rb.loc[x, "p_boot"],
                   "within": rw.loc[x, "coef"], "within_se": rw.loc[x, "se"], "within_p": rw.loc[x, "p"],
                   "within_pboot": rw.loc[x, "p_boot"], "n": rb.attrs["n"], "clusters": rb.attrs["clusters"]})


pf_pair("Baseline", wy)
pf_pair("Alternative capital stock series", wy, k="ln_k_alt")
pf_pair("Persons employed instead of hours", wy, lab="ln_l")
pf_pair("Spending deflated with aggregate GVA deflator", wy, x="ln_gastot_gdpdefl")
pf_pair("Excluding wholesale and retail trade", wy[~wy.klems.isin(["430", "460"])])
pf_pair("Excluding basic metals (331)", wy[wy.klems != "331"])
pf_pair("Errors clustered by two-digit sector", wy, g="sector2")
ROB_PF = pd.DataFrame(rob_pf)
ROB_PF.to_csv(OUT / "tabla_robustez_pf.csv", index=False)
log("\nRobustness, production function (total intangible spending)\n", ROB_PF.round(4).to_string(index=False))

# Pesaran CD test on TWFE residuals (cross-sectional dependence)
d3 = wd.dropna(subset=["ln_va"] + X3)
beta, u = ols(d3.ln_va.to_numpy(), d3[X3 + ["ln_k", "ln_h"]].to_numpy())
d3 = d3.assign(u=u)
U = d3.pivot(index="year", columns="klems", values="u")
R = U.corr().to_numpy()
N, T = U.shape[1], U.shape[0]
iu = np.triu_indices(N, 1)
CD = np.sqrt(2 * T / (N * (N - 1))) * np.nansum(R[iu])
log(f"\nPesaran CD on two-way FE residuals: CD = {CD:.2f}, p = {2 * stats.norm.sf(abs(CD)):.3f} (N={N}, T={T})")
TABLES["A_cd_test"] = pd.DataFrame({"CD": [CD], "p": [2 * stats.norm.sf(abs(CD))], "N": [N], "T": [T]})

# --------------------------------------------- B. long differences / local projections
base = pk[pk.year.between(2016, 2018)].groupby("klems").agg(
    va=("va_real", "mean"), gastot=("gastot_real", "mean"), rd=("rd_real", "mean"),
    sw=("sw_real", "mean"), ip=("ip_real", "mean"), broad=("broad", "first"),
    sector2=("sector2", "first"), subsectors=("subsectors", "first"))
for v in ["gastot", "rd", "sw", "ip"]:
    base[v + "_int"] = base[v] / base.va
lv = pk.pivot(index="klems", columns="year", values=["va_real", "ln_tfp", "k_real", "hours", "va_cur"])
base["pre_va"] = np.log(lv["va_real"][2015] / lv["va_real"][2012])
base["pre_tfp"] = lv["ln_tfp"][2015] - lv["ln_tfp"][2012]
base["ln_va18"] = np.log(lv["va_real"][2018])
# primary activities (only subsector 115) are grouped with mining, utilities and construction
for b in ["manufacturing", "services"]:
    base["b_" + b] = (base.broad == b).astype(float)
BFE = ["b_manufacturing", "b_services"]
log("\nIntangible intensity (R/Y), 2016-2018 means, 64 activities:\n",
    base[["gastot_int", "rd_int", "sw_int", "ip_int"]].describe().round(4).to_string())
log("Top intensities:\n", base.sort_values("gastot_int", ascending=False)[["subsectors", "gastot_int"]].head(8).to_string())

lp_rows = []
for h in range(1, 7):
    y1 = 2018 + h
    base[f"dva_{h}"] = np.log(lv["va_real"][y1] / lv["va_real"][2018])
    base[f"dtfp_{h}"] = lv["ln_tfp"][y1] - lv["ln_tfp"][2018]
    base[f"dk_{h}"] = np.log(lv["k_real"][y1] / lv["k_real"][2018])
    base[f"dh_{h}"] = np.log(lv["hours"][y1] / lv["hours"][2018])

# winsorised intensity (commerce software outlier)
cap = base.gastot_int.quantile(0.95)
base["gastot_int_w"] = base.gastot_int.clip(upper=cap)


def lp(outcome, reg, extra=(), sample=None, h_range=range(1, 7), weights=None, label=""):
    out = []
    for h in h_range:
        d = base if sample is None else base[sample]
        xs = [reg] + list(extra) + BFE
        xs = [x.format(h=h) for x in xs]
        r = fit(d.reset_index(), outcome.format(h=h), xs, se="hc3", boot=[reg] if h in (1, 6) else ())
        sd = d[reg].std()
        out.append({"spec": label, "h": h, "year": 2018 + h, "coef": r.loc[reg, "coef"],
                    "effect_1sd": r.loc[reg, "coef"] * sd, "ci_1sd_lo": (r.loc[reg, "coef"] - 1.96 * r.loc[reg, "se"]) * sd,
                    "ci_1sd_hi": (r.loc[reg, "coef"] + 1.96 * r.loc[reg, "se"]) * sd,
                    "se": r.loc[reg, "se"], "p": r.loc[reg, "p"], "p_boot": r.loc[reg, "p_boot"],
                    "n": r.attrs["n"], "r2": r.attrs["r2"]})
    return pd.DataFrame(out)


specs = [
    ("dva_{h}", "gastot_int", ["dk_{h}", "dh_{h}", "pre_va"], None, "GVA, total intangibles, +K,H,pre-trend"),
    ("dtfp_{h}", "gastot_int", ["pre_tfp"], None, "TFP, total intangibles, +pre-trend"),
    ("dtfp_{h}", "gastot_int_w", ["pre_tfp"], None, "TFP, winsorised intensity"),
    ("dva_{h}", "gastot_int_w", ["dk_{h}", "dh_{h}", "pre_va"], None, "GVA, winsorised intensity"),
    ("dtfp_{h}", "gastot_int", ["pre_tfp"], ~base.index.isin(["430", "460"]), "TFP, excl. commerce"),
    ("dtfp_{h}", "rd_int", ["pre_tfp"], None, "TFP, R&D only"),
    ("dtfp_{h}", "sw_int", ["pre_tfp"], None, "TFP, software only"),
    ("dtfp_{h}", "ip_int", ["pre_tfp"], None, "TFP, patents/IP only"),
    ("dva_{h}", "rd_int", ["dk_{h}", "dh_{h}", "pre_va"], None, "GVA, R&D only"),
]
LP = pd.concat([lp(*s[:2], extra=s[2], sample=s[3], label=s[4]) for s in specs])
LP.to_csv(OUT / "tabla_lp.csv", index=False)
log("\nLocal projections (coef = pesos of value added per peso of spending)\n",
    LP.round(4).to_string(index=False))

# local projections with errors clustered by two-digit sector (input-output neighbours share a sector)
lp_cl = []
for h in (1, 6):
    for y, x, ctr, lab in [(f"dtfp_{h}", "gastot_int_w", ["pre_tfp"], "TFP, total (winsorised)"),
                           (f"dva_{h}", "gastot_int_w", [f"dk_{h}", f"dh_{h}", "pre_va"], "GVA, total (winsorised)"),
                           (f"dtfp_{h}", "rd_int", ["pre_tfp"], "TFP, R&D")]:
        r = fit(base.reset_index(), y, [x] + ctr + BFE, g="sector2", boot=[x])
        lp_cl.append({"spec": lab, "h": h, "coef": r.loc[x, "coef"], "se": r.loc[x, "se"], "p": r.loc[x, "p"],
                      "p_boot": r.loc[x, "p_boot"], "clusters": r.attrs["clusters"]})
LP_CL = pd.DataFrame(lp_cl)
LP_CL.to_csv(OUT / "tabla_lp_cluster_sector.csv", index=False)
log("\nLocal projections, errors clustered by two-digit sector\n", LP_CL.round(4).to_string(index=False))

# joint model with the three components, h = 6 and h = 1
for h in (1, 6):
    keep(f"B_joint_tfp_h{h}", fit(base.reset_index(), f"dtfp_{h}", ["rd_int", "sw_int", "ip_int", "pre_tfp"] + BFE,
                                  se="hc3", boot=["rd_int", "sw_int", "ip_int"]),
         f"TFP growth 2018-{2018 + h} on three intensities")
    keep(f"B_joint_va_h{h}", fit(base.reset_index(), f"dva_{h}",
                                 ["rd_int", "sw_int", "ip_int", f"dk_{h}", f"dh_{h}", "pre_va"] + BFE,
                                 se="hc3", boot=["rd_int", "sw_int", "ip_int"]),
         f"GVA growth 2018-{2018 + h} on three intensities")

# placebo: pre-period growth (2010-2015) should not respond to 2016-2018 spending
base["placebo_tfp"] = lv["ln_tfp"][2015] - lv["ln_tfp"][2010]
base["placebo_va"] = np.log(lv["va_real"][2015] / lv["va_real"][2010])
keep("B_placebo_tfp", fit(base.reset_index(), "placebo_tfp", ["gastot_int", "rd_int"] + BFE, se="hc3",
                          boot=["gastot_int", "rd_int"]), "placebo: TFP growth 2010-2015")
keep("B_placebo_va", fit(base.reset_index(), "placebo_va", ["gastot_int", "rd_int"] + BFE, se="hc3",
                         boot=["gastot_int", "rd_int"]), "placebo: GVA growth 2010-2015")

# Oster (2019) bounds for the TFP h=6 R&D coefficient
def oster_delta(d, y, x, ctrls, rmax_mult=1.3):
    r0 = fit(d, y, [x], se="hc3")
    r1 = fit(d, y, [x] + ctrls, se="hc3")
    b0, R0 = r0.loc[x, "coef"], r0.attrs["r2"]
    b1, R1 = r1.loc[x, "coef"], r1.attrs["r2"]
    rmax = min(1.0, rmax_mult * R1)
    delta = (b1 * (R1 - R0)) / ((b0 - b1) * (rmax - R1)) if b0 != b1 else np.inf
    beta_star = b1 - 1 * (b0 - b1) * (rmax - R1) / (R1 - R0) if R1 != R0 else b1
    return dict(b_uncontrolled=b0, R2_0=R0, b_controlled=b1, R2_1=R1, Rmax=rmax,
                delta_for_zero=delta, beta_star_delta1=beta_star)


ost = pd.DataFrame([dict(outcome=o, regressor=x, **oster_delta(base.reset_index(), o, x, c))
                    for o, x, c in [("dtfp_6", "rd_int", ["pre_tfp"] + BFE),
                                    ("dtfp_6", "gastot_int", ["pre_tfp"] + BFE),
                                    ("dva_6", "rd_int", ["dk_6", "dh_6", "pre_va"] + BFE)]])
ost.to_csv(OUT / "tabla_oster.csv", index=False)
log("\nOster bounds\n", ost.round(4).to_string(index=False))

# leave-one-out influence, TFP h=6, R&D intensity
loo = []
for k in base.index:
    r = fit(base.drop(index=k).reset_index(), "dtfp_6", ["rd_int", "pre_tfp"] + BFE, se="hc3")
    loo.append((k, r.loc["rd_int", "coef"]))
loo = pd.DataFrame(loo, columns=["dropped", "coef"])
log("\nLeave-one-out (TFP 2018-2024 on R&D intensity): min %.3f (drop %s), max %.3f (drop %s)" % (
    loo.coef.min(), loo.loc[loo.coef.idxmin(), "dropped"], loo.coef.max(), loo.loc[loo.coef.idxmax(), "dropped"]))
TABLES["B_loo"] = loo

# --------------------------------------------------- C. input-output spillovers
Z = pd.read_csv(OUT / "io_weights.csv", index_col=0, dtype={0: str})
Z.index = Z.index.astype(str)
Z.columns = Z.columns.astype(str)
Z = Z.loc[base.index, base.index]
go18 = pk[pk.year == 2018].set_index("klems").go_cur.loc[base.index]
A_sales = Z.div(go18, axis=0)          # share of supplier j's output sold to buyer i
np.fill_diagonal(A_sales.values, 0.0)  # exclude own purchases (already in own R&D)
for v in ["rd", "gastot"]:
    emb = A_sales.T @ base[v]          # R&D embodied in i's domestic purchases
    base[v + "_ind_int"] = emb / base.va
log("\nIndirect (embodied) intensity:\n", base[["rd_ind_int", "gastot_ind_int"]].describe().round(5).to_string())
for h in (1, 3, 6):
    keep(f"C_io_tfp_h{h}", fit(base.reset_index(), f"dtfp_{h}", ["rd_int", "rd_ind_int", "pre_tfp"] + BFE,
                               se="hc3", boot=["rd_int", "rd_ind_int"]),
         f"TFP 2018-{2018 + h}: own vs embodied supplier R&D")
    keep(f"C_io_va_h{h}", fit(base.reset_index(), f"dva_{h}",
                              ["rd_int", "rd_ind_int", f"dk_{h}", f"dh_{h}", "pre_va"] + BFE,
                              se="hc3", boot=["rd_int", "rd_ind_int"]),
         f"GVA 2018-{2018 + h}: own vs embodied supplier R&D")
    keep(f"C_io_tfp_tot_h{h}", fit(base.reset_index(), f"dtfp_{h}",
                                   ["gastot_int_w", "gastot_ind_int", "pre_tfp"] + BFE,
                                   se="hc3", boot=["gastot_int_w", "gastot_ind_int"]),
         f"TFP 2018-{2018 + h}: own vs embodied total intangibles (winsorised own)")

# --------------------------------------------------- D. heterogeneity
HT = {"325", "333", "334", "335", "336"}                          # high and medium-high tech manufacturing
KIS = {"511", "512", "515-519", "517-518", "521", "522", "523", "524", "541"}  # knowledge-intensive services
base["hightech"] = base.index.isin(HT).astype(float)
base["kis"] = base.index.isin(KIS).astype(float)
base["rd_x_ht"] = base.rd_int * base.hightech
base["rd_x_kis"] = base.rd_int * base.kis
keep("D_het_tfp_h6", fit(base.reset_index(), "dtfp_6", ["rd_int", "rd_x_ht", "rd_x_kis", "hightech", "kis", "pre_tfp"] + BFE,
                         se="hc3", boot=["rd_int", "rd_x_ht", "rd_x_kis"]),
     "TFP 2018-2024: R&D return by technology group")
keep("D_het_va_h6", fit(base.reset_index(), "dva_6", ["rd_int", "rd_x_ht", "rd_x_kis", "hightech", "kis",
                                                      "dk_6", "dh_6", "pre_va"] + BFE,
                        se="hc3", boot=["rd_int", "rd_x_ht", "rd_x_kis"]),
     "GVA 2018-2024: R&D return by technology group")

# multiple testing in the subsector-interaction design of the first version
ps = pd.read_csv(OUT / "panel_subsector.csv")
ps = ps[~ps.commerce & ps.gva_scn_real.notna()].copy()
ps["ln_va"] = np.log(ps.gva_scn_real)
pvals = {}
for v in ["rd", "sw", "ip"]:
    ps["x"] = np.arcsinh(ps[v] / 1.0)
    rows = []
    for s, d in ps.groupby("subsector"):
        pass
    # interaction model: ln_va_it = a_i + t_t + sum_s b_s (x_it * 1[s]) ; estimated by within transformation
    dd = ps[["subsector", "year", "ln_va", "x"]].copy()
    dd["ln_va"] = dd.ln_va - dd.groupby("subsector").ln_va.transform("mean")
    dd["x"] = dd.x - dd.groupby("subsector").x.transform("mean")
    dd = dd.join(pd.get_dummies(dd.year, prefix="y", drop_first=True, dtype=float))
    for c in ["y_2017", "y_2018"]:
        dd[c] = dd[c] - dd.groupby("subsector")[c].transform("mean")
    subs = sorted(dd.subsector.unique())
    Xs = np.column_stack([(dd.subsector == s).to_numpy(float) * dd.x.to_numpy() for s in subs]
                         + [dd.y_2017.to_numpy(), dd.y_2018.to_numpy()])
    keepcols = np.abs(Xs).sum(0) > 0
    Xs = Xs[:, keepcols]
    labels = [s for s, k in zip(subs + ["y17", "y18"], keepcols) if k][:-2]
    beta, u = ols(dd.ln_va.to_numpy(), Xs)
    n, k = Xs.shape
    dof = n - k - len(subs)
    sig2 = (u @ u) / dof
    se = np.sqrt(np.diag(sig2 * np.linalg.pinv(Xs.T @ Xs)))[: len(labels)]
    p = 2 * stats.t.sf(np.abs(beta[: len(labels)] / se), dof)
    pvals[v] = pd.Series(p, index=labels)


def bh(p, q=0.10):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:kmax]] = True
    return out


mt = []
for v, p in pvals.items():
    mt.append({"spending": v, "tests": len(p), "p<0.05": int((p < 0.05).sum()),
               "expected_false_at_5pct": round(0.05 * len(p), 1),
               "BH_q10_survivors": int(bh(p.values).sum()),
               "Bonferroni_5pct": int((p < 0.05 / len(p)).sum())})
mt = pd.DataFrame(mt)
mt.to_csv(OUT / "tabla_multiple_testing.csv", index=False)
log("\nMultiple testing, subsector-interaction design (70 non-commerce subsectors, classical SEs)\n",
    mt.to_string(index=False))


# ---------------------------------------- E. R&D embodiment multipliers (Leontief)
A = Z.div(go18, axis=1).to_numpy()          # a_ij: input from j... rows supplier, cols buyer
Linv = np.linalg.inv(np.eye(len(A)) - A)
emb_rows = []
y18 = pk[pk.year == 2018].set_index("klems").loc[base.index]
for v in ["rd", "gastot"]:
    r = (y18[v] / y18.go_cur).to_numpy()      # direct spending per peso of gross output
    total = r @ Linv                          # direct + indirect content per peso of final demand
    emb_rows.append(pd.DataFrame({"klems": base.index, "spending": v, "direct": r, "total": total,
                                  "multiplier": np.where(r > 0, total / np.where(r > 0, r, 1), np.nan),
                                  "indirect_share": 1 - r / total}))
emb = pd.concat(emb_rows)
emb = emb.merge(base[["subsectors"]], left_on="klems", right_index=True)
emb.to_csv(OUT / "tabla_multiplicadores_io.csv", index=False)
for v in ["rd", "gastot"]:
    e = emb[emb.spending == v]
    log(f"\nEmbodiment ({v}): median multiplier {e.multiplier.median():.2f}, "
        f"median indirect share {e.indirect_share.median():.2f}, "
        f"share of activities where indirect > direct: {(e.indirect_share > 0.5).mean():.2f}")
    log(e.sort_values("total", ascending=False).head(10)[["klems", "subsectors", "direct", "total", "multiplier", "indirect_share"]].round(5).to_string(index=False))

# ------------------------------------------------------------- save
base.to_csv(OUT / "cross_section_klems.csv")
with pd.ExcelWriter(OUT / "tablas_resultados.xlsx") as xw:
    for k, v in TABLES.items():
        v.to_excel(xw, sheet_name=k[:31])
    LP.to_excel(xw, sheet_name="local_projections", index=False)
    ost.to_excel(xw, sheet_name="oster", index=False)
    mt.to_excel(xw, sheet_name="multiple_testing", index=False)
    emb.to_excel(xw, sheet_name="io_multipliers", index=False)
    ROB_PF.to_excel(xw, sheet_name="robustness_pf", index=False)
    LP_CL.to_excel(xw, sheet_name="lp_cluster_sector", index=False)
    desc.to_excel(xw, sheet_name="descriptives")
    corr.to_excel(xw, sheet_name="correlations")
(OUT / "log_estimaciones.txt").write_text("\n".join(LOG), encoding="utf-8")
