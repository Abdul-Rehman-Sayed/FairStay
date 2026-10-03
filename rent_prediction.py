import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from rent_features import (LOCALITY, LOW_CARD, MODEL_INPUTS, NUMERIC, FrequencyEncoder,
                           feature_family, inr, prepare, split_feature_name)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
IQR_K = 3.0
RENT_FLOOR = 1_000
VOTING_WEIGHTS = [2, 1, 2, 3]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = prepare(raw.drop_duplicates())
    q1, q3 = df["Rent"].quantile([0.25, 0.75])
    cutoff = q3 + IQR_K * (q3 - q1)
    df = df[df["Rent"].between(RENT_FLOOR, cutoff)].reset_index(drop=True)
    log = {"rows_raw": len(raw), "nulls": int(raw.isna().sum().sum()),
           "duplicates": int(raw.duplicated().sum()), "q1": float(q1), "q3": float(q3),
           "upper_cutoff": float(cutoff), "rows_clean": len(df),
           "rows_removed": len(raw) - len(df)}
    return df, log


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("freq", Pipeline([("count", FrequencyEncoder()), ("scale", StandardScaler())]),
         [LOCALITY]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD),
        ("loc", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                              max_categories=80, sparse_output=False), [LOCALITY]),
    ])


def base_learners() -> dict:
    return {
        "lr": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=2,
                                    n_jobs=-1, random_state=RANDOM_STATE),
        "gb": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                        subsample=0.9, random_state=RANDOM_STATE),
        "xgb": XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=6,
                            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                            tree_method="hist", importance_type="gain",
                            n_jobs=-1, random_state=RANDOM_STATE),
    }


def build_models() -> dict:
    b = base_learners()
    return {
        "Linear Regression": b["lr"],
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5,
                                               random_state=RANDOM_STATE),
        "Random Forest": b["rf"],
        "Gradient Boosting": b["gb"],
        "XGBoost": b["xgb"],
        "Voting Ensemble": VotingRegressor(list(base_learners().items()),
                                           weights=VOTING_WEIGHTS),
        "Stacking Ensemble": StackingRegressor(list(base_learners().items()),
                                               final_estimator=RidgeCV(), cv=3),
    }


def metrics(y_true_log, y_pred_log) -> dict:
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
    }


BLUE, BLUE_LIGHT, RED = "#2a78d6", "#9ec5f4", "#e34948"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11, "axes.titlesize": 12,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": INK_2, "ytick.color": INK_2,
    "savefig.dpi": 200, "savefig.bbox": "tight", "mathtext.fontset": "stix",
})
THOUSANDS = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
RUPEES = FuncFormatter(lambda v, _: inr(v))


def save(fig, out_dir: Path, name: str) -> str:
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    return f"{name}.png"


def label_bars(ax, bars, labels):
    for b, text in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), text,
                ha="center", va="bottom", fontsize=9, color=INK_2)


def fig_rent_distribution(raw, cutoff, out_dir):
    rent, p99 = raw["Rent"], raw["Rent"].quantile(0.99)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.hist(rent[rent <= p99], bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    a.axvline(cutoff, color=RED, linewidth=1.2)
    a.text(cutoff, a.get_ylim()[1] * 0.95, f"  IQR cut-off\n  Rs. {inr(cutoff)}", va="top", fontsize=10)
    a.text(0.99, 0.6, f"x-axis cut at 99th percentile\n(Rs. {inr(p99)}); max Rs. {inr(rent.max())}",
           transform=a.transAxes, ha="right", fontsize=9, color=INK_2)
    a.xaxis.set_major_formatter(THOUSANDS)
    a.set(xlabel="Monthly rent (Rs. thousand)", ylabel="Number of listings",
          title=f"(a) Original rent, skewness = {rent.skew():.2f}")
    log_rent = np.log1p(rent)
    b.hist(log_rent, bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    b.axvline(np.log1p(cutoff), color=RED, linewidth=1.2)
    b.set(xlabel="log(1 + rent)", ylabel="Number of listings",
          title=f"(b) Log-transformed rent, skewness = {log_rent.skew():.2f}")
    fig.tight_layout()
    return save(fig, out_dir, "fig_2_1_rent_distribution")


def fig_city_boxplot(raw, out_dir):
    order = raw.groupby("City")["Rent"].median().sort_values(ascending=False).index
    counts = raw["City"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.boxplot([raw.loc[raw["City"] == c, "Rent"] for c in order], widths=0.55, patch_artist=True,
               boxprops=dict(facecolor=BLUE_LIGHT, edgecolor=BLUE),
               medianprops=dict(color=INK, linewidth=1.6),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
               flierprops=dict(marker="o", markersize=2.5, markerfacecolor=MUTED,
                               markeredgecolor="none", alpha=0.6))
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1), [f"{c}\n(n = {counts[c]})" for c in order])
    ax.yaxis.set_major_formatter(RUPEES)
    ax.set_ylabel("Monthly rent (Rs., log scale)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig_2_2_city_boxplot")


def fig_correlation(df, out_dir):
    cols = ["BHK", "Size", "Bathroom", "Floor_No", "Total_Floors", "Rent", "Log_Rent"]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap=LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE]),
                   vmin=-1, vmax=1)
    labels = cols[:-1] + ["log(1+Rent)"]
    ax.set_xticks(range(len(cols)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(corr):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson correlation coefficient")
    fig.tight_layout()
    return save(fig, out_dir, "fig_2_3_correlation")


def fig_median_bhk_furnishing(raw, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    order = ["Unfurnished", "Semi-Furnished", "Furnished"]
    panels = [("BHK", None, "BHK", "(a) Median rent vs BHK"),
              ("Furnishing Status", order, "Furnishing status", "(b) Median rent vs furnishing status")]
    for ax, (col, idx, xlabel, title) in zip(axes, panels):
        s = raw.groupby(col)["Rent"].agg(["median", "size"])
        s = s.reindex(idx) if idx else s
        bars = ax.bar(s.index.astype(str), s["median"], color=BLUE, width=0.6)
        label_bars(ax, bars, [f"Rs. {inr(m)}\nn = {n}" for m, n in zip(s["median"], s["size"])])
        ax.set_ylim(0, s["median"].max() * 1.18)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set(xlabel=xlabel, ylabel="Median monthly rent (Rs. thousand)", title=title)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig_2_4_median_bhk_furnishing")


def fig_model_comparison(table, best, out_dir):
    t = table.sort_values("R2")
    colors = [BLUE if m == best else BLUE_LIGHT for m in t.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    a.barh(t.index, t["R2"], color=colors, height=0.6)
    b.barh(t.index, t["RMSE"], color=colors, height=0.6)
    for y, (r2, rmse) in enumerate(zip(t["R2"], t["RMSE"])):
        a.text(r2 + 0.01, y, f"{r2:.3f}", va="center", fontsize=10)
        b.text(rmse + t["RMSE"].max() * 0.01, y, f"Rs. {inr(rmse)}", va="center", fontsize=10)
    a.set(xlim=(0, 1), xlabel="R$^2$ on test set (rupee scale)  - higher is better",
          title="(a) Coefficient of determination")
    b.set(xlim=(0, t["RMSE"].max() * 1.25), xlabel="RMSE on test set (Rs. thousand)  - lower is better",
          title="(b) Root mean squared error")
    b.xaxis.set_major_formatter(THOUSANDS)
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, BLUE_LIGHT)],
               [f"Selected by cross-validation ({best})", "Other models"],
               loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return save(fig, out_dir, "fig_5_1_model_comparison")


def fig_actual_vs_predicted(y_true, y_pred, name, m, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.scatter(y_true, y_pred, s=10, color=BLUE, alpha=0.45, edgecolors="none")
    xs = np.array([min(y_true.min(), y_pred.min()) * 0.9, max(y_true.max(), y_pred.max()) * 1.1])
    ax.plot(xs, xs, color=INK, linewidth=1.2, label="Perfect prediction (y = x)")
    ax.plot(xs, xs * 1.25, color=MUTED, linewidth=0.9, label="$\\pm$25 % band")
    ax.plot(xs, xs * 0.75, color=MUTED, linewidth=0.9)
    ax.set(xscale="log", yscale="log", xlim=xs, ylim=xs, title=name,
           xlabel="Actual monthly rent (Rs., log scale)",
           ylabel="Predicted monthly rent (Rs., log scale)")
    ax.xaxis.set_major_formatter(RUPEES)
    ax.yaxis.set_major_formatter(RUPEES)
    ax.text(0.03, 0.97, f"R$^2$ = {m['R2']:.3f}\nMAE = Rs. {inr(m['MAE'])}\nMAPE = {m['MAPE']:.1f} %",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.4"))
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    return save(fig, out_dir, "fig_5_2_actual_vs_predicted")


def fig_feature_importance(imp, out_dir):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    for y, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, y, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set(xlim=(0, top.max() * 1.15), xlabel="Share of total gain (XGBoost)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, out_dir, "fig_5_3_feature_importance")


def fig_residuals(y_true, y_pred, name, out_dir):
    resid = y_true - y_pred
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    a.scatter(y_pred, resid, s=10, color=BLUE, alpha=0.45, edgecolors="none")
    a.axhline(0, color=INK, linewidth=1)
    a.set(xscale="log", xlabel="Predicted monthly rent (Rs., log scale)",
          ylabel="Residual = actual - predicted (Rs. thousand)", title="(a) Residuals vs predicted rent")
    a.xaxis.set_major_formatter(RUPEES)
    a.yaxis.set_major_formatter(THOUSANDS)
    b.hist(resid, bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    b.axvline(0, color=INK, linewidth=1)
    b.xaxis.set_major_formatter(THOUSANDS)
    b.set(xlabel="Residual (Rs. thousand)", ylabel="Number of test listings",
          title="(b) Distribution of residuals")
    fig.suptitle(name, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, out_dir, "fig_5_4_residuals")


def main() -> None:
    ap = argparse.ArgumentParser(description="House rent price prediction using ensemble learning")
    ap.add_argument("--data", default=ROOT / "data" / "House_Rent_Dataset.csv", type=Path)
    ap.add_argument("--out", default=ROOT / "outputs", type=Path)
    args = ap.parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    print(f"Loaded {args.data.name}: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"SHA-256: {hashlib.sha256(args.data.read_bytes()).hexdigest()}")

    df, prep_log = clean(raw)
    print(f"Rows after cleaning: {prep_log['rows_clean']} ({prep_log['rows_removed']} removed; "
          f"rent cut-off Rs. {inr(prep_log['upper_cutoff'])})")

    figures = {
        "2.1": fig_rent_distribution(raw, prep_log["upper_cutoff"], fig_dir),
        "2.2": fig_city_boxplot(raw, fig_dir),
        "2.3": fig_correlation(prepare(raw).assign(Log_Rent=np.log1p(raw["Rent"])), fig_dir),
        "2.4": fig_median_bhk_furnishing(raw, fig_dir),
    }

    X, y = df[MODEL_INPUTS], np.log1p(df["Rent"].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, preds = {}, {}, {}
    for name, model in build_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring="r2")
        pipe.fit(X_train, y_train)
        preds[name] = pipe.predict(X_test)
        rows[name] = {**metrics(y_test, preds[name]),
                      "CV_R2_mean": float(cv.mean()), "CV_R2_std": float(cv.std())}
        fitted[name] = pipe
        m = rows[name]
        print(f"  {name:<18} R2={m['R2']:.3f}  R2(log)={m['R2_log']:.3f}  "
              f"RMSE=Rs.{inr(m['RMSE']):>7}  MAE=Rs.{inr(m['MAE']):>6}  "
              f"MAPE={m['MAPE']:.1f}%  CV R2={m['CV_R2_mean']:.3f}+/-{m['CV_R2_std']:.3f}")

    table = pd.DataFrame(rows).T.sort_values("CV_R2_mean", ascending=False)
    best = table.index[0]
    print(f"Selected model (highest mean CV R2): {best}")
    table.to_csv(args.out / "model_comparison.csv", index_label="Model")

    base = ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
    stack = fitted["Stacking Ensemble"].named_steps["model"].final_estimator_
    xgb_pipe = fitted["XGBoost"]
    feature_names = list(xgb_pipe.named_steps["prep"].get_feature_names_out())
    xgb_model = xgb_pipe.named_steps["model"]
    imp = pd.Series(xgb_model.feature_importances_,
                    index=[f"{c} = {lvl}" if lvl else c
                           for c, lvl in map(split_feature_name, feature_names)]
                    ).sort_values(ascending=False)

    y_true_rs, y_best_rs = np.expm1(y_test), np.expm1(preds[best])
    figures.update({
        "5.1": fig_model_comparison(table, best, fig_dir),
        "5.2": fig_actual_vs_predicted(y_true_rs, y_best_rs, best, rows[best], fig_dir),
        "5.3": fig_feature_importance(imp, fig_dir),
        "5.4": fig_residuals(y_true_rs, y_best_rs, best, fig_dir),
    })

    MODEL_DIR.mkdir(exist_ok=True)
    files = {name: name.lower().replace(" ", "_") + ".joblib" for name in fitted}
    for name, pipe in fitted.items():
        joblib.dump(pipe, MODEL_DIR / files[name], compress=3)

    contrib = xgb_model.get_booster().predict(
        xgboost.DMatrix(xgb_pipe.named_steps["prep"].transform(X_test)), pred_contribs=True)
    shap_share = (pd.DataFrame(np.abs(contrib[:, :-1]), columns=map(feature_family, feature_names))
                  .T.groupby(level=0).sum().T.mean())
    shap_share = (shap_share / shap_share.sum()).sort_values(ascending=False)

    counts = X_train.groupby(["City", LOCALITY]).size()
    app_meta = {
        "selected_model": best,
        "models": {n: {"file": files[n], **table.loc[n].to_dict()} for n in table.index},
        "n_clean": len(df), "n_test": len(X_test),
        "rent_cutoff": prep_log["upper_cutoff"], "voting_weights": VOTING_WEIGHTS,
        "localities": {c: [[loc, int(n)] for loc, n in counts.loc[c].sort_values(ascending=False).items()]
                       for c in sorted(X_train["City"].unique())},
        "options": {c: [v for v, n in X_train[c].value_counts().items() if n >= 10] for c in LOW_CARD},
        "default_posted_month": int(X_train["Posted_Month"].mode()[0]),
        "limits": {c: int(df[c].max()) for c in ["BHK", "Bathroom", "Size", "Total_Floors"]},
        "global_importance": shap_share.astype(float).round(4).to_dict(),
        "sklearn_version": sklearn.__version__, "xgboost_version": xgboost.__version__,
    }
    (args.out / "app_metadata.json").write_text(json.dumps(app_meta, indent=2))

    results = {
        "preprocessing": prep_log,
        "split": {"train": len(X_train), "test": len(X_test)},
        "models": table.to_dict(orient="index"),
        "best_model": best,
        "stacking_weights": dict(zip(base, map(float, stack.coef_))),
        "residual_corr_log": pd.DataFrame({k: y_test - preds[k] for k in base}).corr().round(4).to_dict(),
        "xgb_importance_top15": imp.head(15).astype(float).to_dict(),
        "figures": figures,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {len(fitted)} models to {MODEL_DIR} and results, metadata and "
          f"{len(figures)} figures to {args.out}")


if __name__ == "__main__":
    main()
