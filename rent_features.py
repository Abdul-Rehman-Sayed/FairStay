import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FLOOR_WORDS = {"Ground": 0, "Upper Basement": -1, "Lower Basement": -2}

NUMERIC = ["BHK", "Size", "Bathroom", "Floor_No", "Total_Floors", "Floor_Ratio",
           "Size_per_BHK", "Bath_per_BHK", "Rooms_Total", "Log_Size", "Posted_Month"]
LOW_CARD = ["Area Type", "City", "Furnishing Status", "Tenant Preferred", "Point of Contact"]
LOCALITY = "Area Locality"
MODEL_INPUTS = NUMERIC + [LOCALITY] + LOW_CARD

FAMILY = {
    "BHK": "BHK", "Size": "Size", "Log_Size": "Size", "Bathroom": "Bathrooms",
    "Size_per_BHK": "Room mix", "Bath_per_BHK": "Room mix", "Rooms_Total": "Room mix",
    "Floor_No": "Floor", "Total_Floors": "Floor", "Floor_Ratio": "Floor",
    "Posted_Month": "Listing month", "Locality_Freq": "Locality", "Area Locality": "Locality",
    "Area Type": "Area type", "City": "City", "Furnishing Status": "Furnishing",
    "Tenant Preferred": "Tenant preferred", "Point of Contact": "Point of contact",
}


def inr(value: float) -> str:
    sign, digits = ("-" if value < 0 else ""), str(abs(int(round(value))))
    head, tail = digits[:-3], digits[-3:]
    groups = [head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1]
    return sign + ",".join(groups + [tail])


def parse_floor(text: str) -> tuple[int, float]:
    level, _, total = str(text).partition(" out of ")
    level = level.strip()
    floor_no = FLOOR_WORDS[level] if level in FLOOR_WORDS else int(level)
    return floor_no, float(total) if total else np.nan


def floor_text(floor_no: int, total: int) -> str:
    word = {v: k for k, v in FLOOR_WORDS.items()}.get(floor_no, str(floor_no))
    return f"{word} out of {total}"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = df["Floor"].apply(parse_floor)
    df["Floor_No"] = parsed.str[0].astype(int)
    df["Total_Floors"] = parsed.str[1].fillna(df["Floor_No"].clip(lower=0))
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].where(df["Total_Floors"] > 0)).fillna(0.0)
    df["Posted_Month"] = pd.to_datetime(df["Posted On"]).dt.month
    df["Size_per_BHK"] = df["Size"] / df["BHK"]
    df["Bath_per_BHK"] = df["Bathroom"] / df["BHK"]
    df["Rooms_Total"] = df["BHK"] + df["Bathroom"]
    df["Log_Size"] = np.log1p(df["Size"])
    return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.counts_ = pd.Series(np.asarray(X).ravel()).value_counts()
        return self

    def transform(self, X):
        col = pd.Series(np.asarray(X).ravel())
        return col.map(self.counts_).fillna(0).to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["Locality_Freq"])


def split_feature_name(name: str) -> tuple[str, str]:
    branch, feat = name.split("__", 1)
    if branch in ("num", "freq"):
        return feat, ""
    col = next(c for c in LOW_CARD + [LOCALITY] if feat.startswith(c + "_"))
    level = feat[len(col) + 1:]
    return col, "(rare localities)" if level == "infrequent_sklearn" else level


def feature_family(name: str) -> str:
    return FAMILY[split_feature_name(name)[0]]
