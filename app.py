import calendar
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import xgboost

from rent_features import MODEL_INPUTS, feature_family, floor_text, prepare
from rent_features import inr as _inr

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

THIN_EVIDENCE = 10

st.set_page_config(
    page_title="FairStay - fair rent estimates for Indian metros",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

:root{
  --paper:#FBFAF7; --surface:#FFFFFF;
  --ink:#14161A; --ink-2:#565C66; --ink-3:#8B9099;
  --rule:#E4E0D8; --rule-strong:#CFCABE;
  --accent:#14544A; --accent-muted:#A8BCB6; --track:#EDEAE2;
  --pos:#1B6B4F; --neg:#A63A25;
  --serif:'Newsreader',Georgia,'Times New Roman',serif;
  --sans:'Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
}

[data-testid="stAppViewContainer"]{background:var(--paper);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppDeployButton"],[data-testid="stStatusWidget"]{display:none;}
[data-testid="stMainBlockContainer"]{max-width:880px;padding:3.4rem 2rem 4rem;}
[data-testid="stSidebar"]{background:#F4F2EB;border-right:1px solid var(--rule);
  width:352px!important;min-width:352px!important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:2.1rem 1.5rem 2rem;}
[data-testid="stForm"]{border:none;padding:0;}
[data-testid="stWidgetLabel"] p{font-family:var(--sans);font-size:.68rem;font-weight:500;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);}
.stSelectbox div[data-baseweb="select"]>div,
[data-testid="stNumberInputContainer"]{background:var(--surface);border-color:var(--rule-strong);}
[data-testid="stNumberInputContainer"] div[data-baseweb="input"],
[data-testid="stNumberInputContainer"] div[data-baseweb="base-input"],
[data-testid="stNumberInputField"],
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"]{background:transparent;}
[data-testid="stNumberInputStepUp"]:hover,
[data-testid="stNumberInputStepDown"]:hover{background:var(--track);color:var(--ink);}
.stFormSubmitButton{width:100%;}
.stFormSubmitButton button{width:100%;background:var(--ink);border:1px solid var(--ink);
  color:#FBFAF7;font-family:var(--sans);font-size:.72rem;font-weight:500;
  letter-spacing:.11em;text-transform:uppercase;padding:.7rem 1rem;}
.stFormSubmitButton button:hover,
.stFormSubmitButton button:focus:not(:active){background:var(--accent);border-color:var(--accent);color:#fff;}

.masthead{margin:0 0 1.4rem;}
.eyebrow{font-family:var(--sans);font-size:.68rem;font-weight:500;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 .85rem;}
.masthead h1{font-family:var(--serif);font-size:3.4rem;line-height:1;font-weight:400;
  letter-spacing:-.015em;color:var(--ink);margin:0 0 .85rem;}
.lede{font-family:var(--sans);font-size:1rem;line-height:1.62;color:var(--ink-2);
  max-width:46em;margin:0;}
.lede em{font-style:italic;color:var(--ink);}

.section{display:flex;align-items:center;gap:1rem;margin:2.6rem 0 1.25rem;}
.section h2{font-family:var(--sans);font-size:.72rem;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink);margin:0;white-space:nowrap;}
.section .line{flex:1;height:1px;background:var(--rule);}
.section .aside{font-family:var(--sans);font-size:.72rem;color:var(--ink-3);white-space:nowrap;}

.figure{display:flex;align-items:baseline;gap:.9rem;flex-wrap:wrap;margin:0 0 .35rem;}
.figure__value{font-family:var(--serif);font-size:3.9rem;line-height:1;letter-spacing:-.02em;
  color:var(--ink);font-variant-numeric:tabular-nums;}
.figure__unit{font-family:var(--sans);font-size:.85rem;color:var(--ink-3);}
.spec{font-family:var(--sans);font-size:.92rem;line-height:1.55;color:var(--ink-2);margin:0;}
.meter{margin:1.5rem 0 .2rem;}
.meter svg{display:block;width:100%;height:auto;}

.verdict{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--ink);
  padding:1rem 1.2rem;margin:1.7rem 0 0;}
.verdict__label{font-family:var(--sans);font-size:.68rem;font-weight:600;letter-spacing:.12em;
  text-transform:uppercase;margin:0 0 .35rem;}
.verdict__body{font-family:var(--sans);font-size:.94rem;line-height:1.55;color:var(--ink-2);margin:0;}
.verdict--under{border-left-color:var(--pos);} .verdict--under .verdict__label{color:var(--pos);}
.verdict--over{border-left-color:var(--neg);} .verdict--over .verdict__label{color:var(--neg);}
.verdict--fair{border-left-color:var(--accent);} .verdict--fair .verdict__label{color:var(--accent);}

.ledger{width:100%;border-collapse:collapse;font-family:var(--sans);font-size:.92rem;}
.ledger th{font-size:.66rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;padding:0 0 .55rem;border-bottom:1px solid var(--rule-strong);}
.ledger td{padding:.68rem 0;border-bottom:1px solid var(--rule);color:var(--ink);
  font-variant-numeric:tabular-nums;}
.ledger tbody tr:last-child td{border-bottom:none;}
.ledger .num{text-align:right;padding-left:1.1rem;}
.ledger .model{font-weight:500;}
.ledger .tag{font-size:.64rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent);margin-left:.55rem;}
.ledger tr.selected td{background:#F1EEE6;}

.bars{display:grid;grid-template-columns:8.5rem 1fr 3rem;gap:.6rem 1rem;align-items:center;padding-bottom:.4rem;
  font-family:var(--sans);font-size:.86rem;}
.bars__name{color:var(--ink-2);}
.bars__track{height:9px;background:var(--track);}
.bars__fill{height:9px;background:var(--accent);}
.bars__fill--minor{background:var(--accent-muted);}
.bars__value{text-align:right;color:var(--ink);font-variant-numeric:tabular-nums;}

.wf{display:grid;grid-template-columns:11.5rem 1fr 3.4rem;gap:.6rem 1rem;align-items:center;
  font-family:var(--sans);font-size:.86rem;}
.wf__name{color:var(--ink);line-height:1.35;}
.wf__name span{display:block;color:var(--ink-3);font-size:.8rem;}
.wf__track{position:relative;height:11px;background:var(--track);}
.wf__track::before{content:"";position:absolute;left:50%;top:-4px;bottom:-4px;width:1px;
  background:var(--rule-strong);}
.wf__bar{position:absolute;top:0;height:11px;}
.wf__bar--up{background:var(--accent);}
.wf__bar--down{background:var(--neg);}
.wf__value{text-align:right;color:var(--ink);font-variant-numeric:tabular-nums;}
.wf__ends{display:flex;justify-content:space-between;gap:1.5rem;flex-wrap:wrap;
  font-family:var(--sans);font-size:.82rem;color:var(--ink-3);border-top:1px solid var(--rule);
  margin-top:1rem;padding-top:.75rem;}
.wf__ends strong{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums;}

.note{font-family:var(--sans);font-size:.86rem;line-height:1.6;color:var(--ink-2);
  border-left:2px solid var(--rule-strong);padding-left:.9rem;margin:1.2rem 0 0;}
.note strong{color:var(--ink);font-weight:500;}
.caption{font-family:var(--sans);font-size:.84rem;line-height:1.6;color:var(--ink-3);
  max-width:54em;margin:1.3rem 0 0;}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1.2rem;padding:1.15rem 0;
  border-top:1px solid var(--rule-strong);border-bottom:1px solid var(--rule-strong);margin:0 0 2.2rem;}
.stats__n{font-family:var(--serif);font-size:1.65rem;line-height:1;color:var(--ink);
  font-variant-numeric:tabular-nums;}
.stats__k{font-family:var(--sans);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-top:.45rem;}
.steps{display:grid;gap:1.15rem;margin:0;}
.step{display:grid;grid-template-columns:2.2rem 1fr;align-items:baseline;}
.step__n{font-family:var(--serif);font-size:1rem;color:var(--accent);}
.step__t{font-family:var(--sans);font-size:.93rem;line-height:1.62;color:var(--ink-2);}
.step__t strong{color:var(--ink);font-weight:500;}

.side-head{font-family:var(--sans);font-size:.7rem;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink);margin:0 0 .35rem;}
.side-sub{font-family:var(--sans);font-size:.82rem;line-height:1.55;color:var(--ink-2);margin:0 0 1.3rem;}
.side-foot{font-family:var(--sans);font-size:.74rem;line-height:1.6;color:var(--ink-3);
  border-top:1px solid var(--rule);padding-top:.9rem;margin-top:1.5rem;}

.colophon{display:flex;justify-content:space-between;gap:2rem;flex-wrap:wrap;
  border-top:1px solid var(--rule);margin-top:3.4rem;padding-top:1rem;
  font-family:var(--sans);font-size:.76rem;line-height:1.6;color:var(--ink-3);}

@media (max-width:640px){
  .masthead h1{font-size:2.6rem;}
  .figure__value{font-size:3rem;}
  .stats{grid-template-columns:repeat(2,1fr);row-gap:1.3rem;}
  .bars{grid-template-columns:6.5rem 1fr 2.8rem;}
}
</style>
"""


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inr(value: float) -> str:
    return "₹" + _inr(value)


def section(title: str, aside: str = "") -> None:
    right = f'<span class="aside">{aside}</span>' if aside else ""
    html(f'<div class="section"><h2>{title}</h2><span class="line"></span>{right}</div>')


@st.cache_data
def load_metadata() -> dict:
    return json.loads((OUT_DIR / "app_metadata.json").read_text())


@st.cache_resource
def load_models(files: tuple[tuple[str, str], ...]) -> dict:
    return {name: joblib.load(MODEL_DIR / f) for name, f in files}


def check_versions(metadata: dict) -> None:
    for lib, key, running in [("scikit-learn", "sklearn_version", sklearn.__version__),
                              ("xgboost", "xgboost_version", xgboost.__version__)]:
        if metadata[key] != running:
            st.error(f"Models were trained with {lib} {metadata[key]} but this app runs "
                     f"{running}. Install {lib}=={metadata[key]} or re-run "
                     "`python rent_prediction.py` with this interpreter.")
            st.stop()


def build_listing(form: dict, posted_month: int) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "Posted On": f"2022-{posted_month:02d}-15",
        "BHK": form["bhk"], "Size": form["size"], "Bathroom": form["bathroom"],
        "Floor": floor_text(form["floor"], form["total_floors"]),
        "Area Type": form["area_type"], "Area Locality": form["locality"],
        "City": form["city"], "Furnishing Status": form["furnishing"],
        "Tenant Preferred": form["tenant"], "Point of Contact": form["contact"],
    }])
    return prepare(raw)[MODEL_INPUTS]


def rent_meter(low: float, point: float, high: float,
               asking: float | None = None, tone: str = "accent") -> str:
    width, pad = 760, 34
    headroom = 46 if asking else 0
    track_y = headroom + 20
    height = track_y + 46
    colour = {"pos": "#1B6B4F", "neg": "#A63A25", "accent": "#14544A"}[tone]

    span = max(high - low, 1.0)
    lo_dom, hi_dom = low - span * 0.45, high + span * 0.45
    if asking:
        lo_dom, hi_dom = min(lo_dom, asking - span * 0.3), max(hi_dom, asking + span * 0.3)
    lo_dom = max(lo_dom, 0.0)

    def x(value: float) -> float:
        t = (value - lo_dom) / (hi_dom - lo_dom)
        return pad + min(max(t, 0.0), 1.0) * (width - 2 * pad)

    def label_x(value: float) -> float:
        return min(max(x(value), pad + 34), width - pad - 34)

    font = 'font-family="Inter, ui-sans-serif, system-ui, sans-serif"'
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Fair rent range {inr(low)} to {inr(high)}, '
        f'best estimate {inr(point)}." xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{pad}" y="{track_y}" width="{width - 2 * pad}" height="10" fill="#EDEAE2"/>',
        f'<rect x="{x(low):.1f}" y="{track_y}" width="{x(high) - x(low):.1f}" '
        f'height="10" fill="#C9DCD5"/>',
    ]
    for edge in (low, high):
        parts.append(f'<rect x="{x(edge):.1f}" y="{track_y - 4}" width="1" '
                     f'height="18" fill="#7C9E95"/>')
    parts.append(f'<rect x="{x(point) - 1.5:.1f}" y="{track_y - 9}" width="3" '
                 f'height="28" fill="#14544A"/>')
    parts.append(f'<text x="{x(low):.1f}" y="{track_y + 34}" text-anchor="middle" '
                 f'{font} font-size="13" fill="#565C66">{inr(low)}</text>')
    parts.append(f'<text x="{x(high):.1f}" y="{track_y + 34}" text-anchor="middle" '
                 f'{font} font-size="13" fill="#565C66">{inr(high)}</text>')

    if asking:
        ax = x(asking)
        parts.append(f'<line x1="{ax:.1f}" y1="24" x2="{ax:.1f}" y2="{track_y - 11}" '
                     f'stroke="{colour}" stroke-width="1" stroke-dasharray="2 3"/>')
        parts.append(f'<polygon points="{ax - 5:.1f},{track_y - 12} {ax + 5:.1f},'
                     f'{track_y - 12} {ax:.1f},{track_y - 1}" fill="{colour}"/>')
        parts.append(f'<text x="{label_x(asking):.1f}" y="16" text-anchor="middle" '
                     f'{font} font-size="13" font-weight="500" fill="{colour}">'
                     f'Asking {inr(asking)}</text>')

    parts.append("</svg>")
    return f'<div class="meter">{"".join(parts)}</div>'


def ledger(preds: dict, metadata: dict) -> str:
    selected = metadata["selected_model"]
    rows = []
    for name, info in metadata["models"].items():
        tag = '<span class="tag">selected</span>' if name == selected else ""
        cls = ' class="selected"' if name == selected else ""
        rows.append(
            f'<tr{cls}><td class="model">{name}{tag}</td>'
            f'<td class="num">{inr(preds[name])}</td>'
            f'<td class="num">{info["CV_R2_mean"]:.3f}</td>'
            f'<td class="num">{inr(info["RMSE"])}</td>'
            f'<td class="num">{inr(info["MAE"])}</td>'
            f'<td class="num">{info["MAPE"]:.1f}%</td></tr>')
    return ('<table class="ledger"><thead><tr><th>Model</th>'
            '<th class="num">Prediction</th><th class="num">CV R²</th>'
            '<th class="num">Test RMSE</th><th class="num">Test MAE</th>'
            '<th class="num">Test MAPE</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def importance_bars(series: pd.Series) -> str:
    peak = float(series.max())
    top = max(peak, 1e-9)
    cells = []
    for name, value in series.items():
        minor = "" if value == peak else " bars__fill--minor"
        cells.append(
            f'<div class="bars__name">{name}</div>'
            f'<div class="bars__track"><div class="bars__fill{minor}" '
            f'style="width:{value / top:.1%}"></div></div>'
            f'<div class="bars__value">{value:.1%}</div>'
        )
    return f'<div class="bars">{"".join(cells)}</div>'


def local_contributions(xgb_pipe, X: pd.DataFrame) -> tuple[float, pd.Series]:
    prep = xgb_pipe.named_steps["prep"]
    booster = xgb_pipe.named_steps["model"].get_booster()
    values = booster.predict(xgboost.DMatrix(prep.transform(X)), pred_contribs=True)[0]
    families = [feature_family(n) for n in prep.get_feature_names_out()]
    grouped = pd.Series(values[:-1]).groupby(families).sum()
    return float(values[-1]), grouped.reindex(grouped.abs().sort_values(ascending=False).index)


def contribution_bars(contributions: pd.Series, chosen: dict) -> str:
    widest = max(float(contributions.abs().max()), 1e-9)
    rows = []
    for field, value in contributions.items():
        share = min(abs(value) / widest, 1.0) * 50
        side = "up" if value >= 0 else "down"
        edge = "left:50%" if value >= 0 else f"left:{50 - share:.2f}%"
        rows.append(
            f'<div class="wf__name">{field}<span>{chosen.get(field, "")}</span></div>'
            f'<div class="wf__track"><div class="wf__bar wf__bar--{side}" '
            f'style="{edge};width:{share:.2f}%"></div></div>'
            f'<div class="wf__value">×{np.exp(value):.2f}</div>'
        )
    return f'<div class="wf">{"".join(rows)}</div>'


def floor_label(floor: int) -> str:
    return {0: "Ground floor", -1: "Upper basement", -2: "Lower basement"}.get(
        floor, f"Floor {floor}")


def listing_breakdown(xgb_pipe, X: pd.DataFrame, form: dict, preds: dict,
                      posted_month: int) -> None:
    base_log, contributions = local_contributions(xgb_pipe, X)
    chosen = {
        "City": form["city"], "Locality": form["locality"],
        "Size": f"{form['size']:,} sq ft", "BHK": f"{form['bhk']} BHK",
        "Bathrooms": str(form["bathroom"]),
        "Room mix": f"{form['size'] / form['bhk']:,.0f} sq ft per bedroom",
        "Floor": f"{floor_label(form['floor'])} of {form['total_floors']}",
        "Area type": form["area_type"], "Furnishing": form["furnishing"],
        "Tenant preferred": form["tenant"],
        "Point of contact": form["contact"].replace("Contact ", ""),
        "Listing month": f"{calendar.month_name[posted_month]} 2022 (fixed)",
    }
    html(contribution_bars(contributions, chosen))
    html('<div class="wf__ends">'
         f'<span>Typical listing in the training data <strong>{inr(np.expm1(base_log))}</strong></span>'
         f'<span>This listing, as XGBoost reads it '
         f'<strong>{inr(preds["XGBoost"])}</strong></span></div>')
    html('<p class="caption">Read each figure as a multiplier on the rent: the '
         "fields carry the typical listing up or down to this one. These are "
         "XGBoost's exact TreeSHAP contributions; they add up to its output in the "
         "log-rent it was trained on, so multiplying them through the typical "
         "listing lands on XGBoost's own prediction. XGBoost carries the largest "
         "weight in the voting ensemble, but the headline above blends it with "
         "three other models, so the two figures differ a little.</p>")


def landing(metadata: dict, mape: float) -> None:
    n_localities = sum(len(v) for v in metadata["localities"].values())
    stats = [
        (f"{metadata['n_clean']:,}", "listings trained on"),
        (f"{len(metadata['localities'])}", "metro cities"),
        (f"{n_localities:,}", "localities priced"),
        (f"±{mape:.1%}", "typical error"),
    ]
    html('<div class="stats">' + "".join(
        f'<div><div class="stats__n">{n}</div><div class="stats__k">{k}</div></div>'
        for n, k in stats) + "</div>")

    w = metadata["voting_weights"]
    section("How the estimate is made")
    steps = [
        ("Seven models are trained on the same listings: linear regression, a "
         "decision tree, a random forest, gradient boosting, XGBoost, and two "
         "ensembles built from them. The one with the best cross-validated score "
         f"on the training data, the <strong>{metadata['selected_model'].lower()}</strong>, "
         "gives the best estimate."),
        (f"The voting ensemble averages linear regression, random forest, gradient "
         f"boosting and XGBoost with weights {w[0]} : {w[1]} : {w[2]} : {w[3]}. A linear "
         "model and tree models make different kinds of mistakes, so averaging them "
         "cancels some of each."),
        (f"The range around the estimate is the selected model's average error on "
         f"{metadata['n_test']:,} held-out listings — <strong>±{mape:.1%}</strong>. "
         "An asking rent, if you enter one, is placed on the same scale."),
    ]
    html('<div class="steps">' + "".join(
        f'<div class="step"><div class="step__n">{i:02d}</div>'
        f'<div class="step__t">{text}</div></div>'
        for i, text in enumerate(steps, start=1)) + "</div>")


def property_form(metadata: dict) -> tuple[bool, dict]:
    options, limits = metadata["options"], metadata["limits"]
    cities = sorted(metadata["localities"])

    html('<p class="side-head">The property</p>'
         '<p class="side-sub">Describe the listing. Every field is one of the '
         'inputs the models were trained on.</p>')

    city = st.selectbox("City", cities,
                        index=cities.index("Mumbai") if "Mumbai" in cities else 0)
    city_localities = metadata["localities"][city]

    with st.form("rent_form"):
        locality = st.selectbox(
            "Locality", [name for name, _ in city_localities],
            help=f"{len(city_localities):,} localities in {city}, most-listed first.")

        col1, col2 = st.columns(2)
        with col1:
            bhk = st.number_input("Bedrooms (BHK)", 1, limits["BHK"], 2, step=1)
        with col2:
            bathroom = st.number_input("Bathrooms", 1, limits["Bathroom"], 2, step=1)

        size = st.number_input("Size (sq ft)", 100, limits["Size"], 900, step=50)
        area_type = st.selectbox(
            "Size measured as", options["Area Type"],
            help="Carpet area is the usable floor area. Super area also counts walls "
                 "and a share of the building's common areas, so it reads larger.")

        col3, col4 = st.columns(2)
        with col3:
            floor = st.number_input("Floor", -2, limits["Total_Floors"], 2, step=1,
                                    help="0 = ground, -1 = upper basement, "
                                         "-2 = lower basement.")
        with col4:
            total_floors = st.number_input("Total floors", 0,
                                           limits["Total_Floors"], 5, step=1)

        furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"],
                                  index=1)
        tenant = st.selectbox("Tenant preferred", options["Tenant Preferred"])
        contact = st.selectbox("Listed by", options["Point of Contact"],
                               format_func=lambda v: v.replace("Contact ", ""))
        asking_rent = st.number_input(
            "Asking rent (₹ per month)", 0, 5_000_000, 0, step=1_000,
            help="Optional. Leave at zero to skip; enter a figure to see where "
                 "it falls against the estimate.",
        )
        submitted = st.form_submit_button("Estimate fair rent", type="primary",
                                          width="stretch")

    html('<p class="side-foot">Seven models trained on an 80/20 split of '
         f'{metadata["n_clean"]:,} listings, scored on the {metadata["n_test"]:,} '
         'they never saw.</p>')

    n_listings = dict(city_localities).get(locality, 0)
    return submitted, dict(city=city, locality=locality, n_listings=n_listings, bhk=bhk,
                           bathroom=bathroom, size=size, area_type=area_type, floor=floor,
                           total_floors=max(total_floors, floor, 0), furnishing=furnishing,
                           tenant=tenant, contact=contact, asking_rent=asking_rent)


def verdict_block(asking: float, point: float, low: float, high: float) -> tuple[str, str]:
    gap = (asking - point) / point
    if asking < low:
        tone, label = "pos", "Below the fair range"
        body = (f"{inr(asking)} is {abs(gap):.0%} under the estimate for a property "
                "like this one. Worth a look, and worth asking why.")
    elif asking > high:
        tone, label = "neg", "Above the fair range"
        body = (f"{inr(asking)} is {gap:.0%} over the estimate. Comparable listings "
                "suggest there is room to negotiate.")
    else:
        tone, label = "accent", "Within the fair range"
        body = (f"{inr(asking)} sits inside the range these models expect for a "
                "property like this one.")
    css = {"pos": "under", "neg": "over", "accent": "fair"}[tone]
    return (f'<div class="verdict verdict--{css}"><p class="verdict__label">{label}</p>'
            f'<p class="verdict__body">{body}</p></div>'), tone


def colophon(metadata: dict) -> None:
    html('<div class="colophon"><span>Linear regression · Decision tree · Random forest '
         '· Gradient boosting · XGBoost · Voting · Stacking — scikit-learn '
         f'{metadata["sklearn_version"]}, XGBoost {metadata["xgboost_version"]}</span>'
         "<span>Asking rents listed April–July 2022, not a live market valuation.</span></div>")


def main() -> None:
    html(CSS)
    metadata = load_metadata()
    check_versions(metadata)
    models = load_models(tuple((n, m["file"]) for n, m in metadata["models"].items()))
    selected = metadata["selected_model"]
    mape = metadata["models"][selected]["MAPE"] / 100
    posted_month = metadata["default_posted_month"]

    with st.sidebar:
        submitted, form = property_form(metadata)

    html('<div class="masthead"><p class="eyebrow">Rent benchmark · '
         f'{len(metadata["localities"])} Indian metros</p>'
         "<h1>FairStay</h1>"
         "<p class=\"lede\">What a flat <em>should</em> rent for, estimated from "
         f"{metadata['n_clean']:,} real listings by an ensemble of machine-learning "
         "models — with the size of the typical error stated up front rather than "
         "buried in a footnote.</p></div>")

    if not submitted:
        landing(metadata, mape)
        colophon(metadata)
        return

    X = build_listing(form, posted_month)
    preds = {name: float(np.expm1(m.predict(X)[0])) for name, m in models.items()}
    point = preds[selected]
    low, high = point * (1 - mape), point * (1 + mape)
    asking = float(form["asking_rent"]) or None

    section("Fair rent", f"{selected.lower()} · range is the estimate ±{mape:.1%}")

    verdict_html, tone = ("", "accent")
    if asking:
        verdict_html, tone = verdict_block(asking, point, low, high)

    html(f'<div class="figure"><span class="figure__value">{inr(point)}</span>'
         '<span class="figure__unit">per month · best estimate</span></div>'
         f'<p class="spec">{form["bhk"]} BHK · {form["bathroom"]} bath · '
         f'{form["size"]:,} sq ft ({form["area_type"].lower()}) · '
         f'{floor_label(form["floor"]).lower()} of {form["total_floors"]} · '
         f'{form["furnishing"].lower()} · {form["locality"]}, {form["city"]} · '
         f'listed by {form["contact"].replace("Contact ", "").lower()}</p>')
    html(rent_meter(low, point, high, asking, tone))

    if verdict_html:
        html(verdict_html)

    if form["n_listings"] < THIN_EVIDENCE:
        html(f'<p class="note"><strong>Thin evidence for {form["locality"]}.</strong> '
             f'The training data holds {form["n_listings"]} listing(s) for that area, so '
             "this estimate leans mostly on the city and the property itself. Treat "
             "it as a rough guide.</p>")
    if high > metadata["rent_cutoff"]:
        html('<p class="note"><strong>Near the top of the training range.</strong> '
             f'Listings above {inr(metadata["rent_cutoff"])} were removed as outliers '
             "before training, so the models have not seen rents this high and will "
             "tend to underestimate them.</p>")

    section("This listing, field by field", "XGBoost · TreeSHAP")
    listing_breakdown(models["XGBoost"], X, form, preds, posted_month)

    spread = (max(preds.values()) - min(preds.values())) / point
    section("Model by model", f"they span {spread:.1%} of the estimate here")
    html(ledger(preds, metadata))
    html('<p class="caption">Models are listed by cross-validated R² on the '
         "training data, which is how the headline model was chosen; the test "
         f"columns are measured on {metadata['n_test']:,} listings none of them saw. "
         "Agreement between the models is not the same as certainty: they learn "
         "from the same listings, so they tend to be wrong together.</p>")

    section("What moves the number", "XGBoost · mean |SHAP| on the test set")
    importance = pd.Series(metadata["global_importance"])
    html(importance_bars(importance))
    html('<p class="caption">How much each input field moves XGBoost&rsquo;s '
         f"prediction on average, across the {metadata['n_test']:,} test listings, "
         "as a share of the total. Where the panel above accounts for this one "
         "listing, this shows how the model weighs inputs in general. Point of "
         "contact ranks high because agents list the pricier flats in every city; "
         "it marks the market segment, it does not set the rent.</p>")

    colophon(metadata)


if __name__ == "__main__":
    main()
