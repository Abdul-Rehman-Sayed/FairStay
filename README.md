# FairStay

**House rent price prediction for Indian metropolitan cities using ensemble learning.**

FairStay predicts the monthly rent of a flat in six Indian metros from its listing
details, compares single models against bagging, boosting, voting and stacking
ensembles, and serves the result as a Streamlit app that estimates a fair rent
range and judges an asking rent against it.

---

## Dataset

[House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
(Sourav Banerjee, Kaggle, 2022): 4,746 rental listings, 12 columns, six cities
(Mumbai, Chennai, Bangalore, Hyderabad, Delhi, Kolkata), posted between April and
July 2022.

Download the CSV and save it as **`data/House_Rent_Dataset.csv`** (it is not
committed). The file this project was built on has SHA-256
`347368b0b20bf19599307d2d23d0072560e08e5cbf03deb9320c474b63343e57`;
`rent_prediction.py` prints the hash of whatever file it loads, so you can confirm
you have the same one.

Columns: `Posted On, BHK, Rent, Size, Floor, Area Type, Area Locality, City,
Furnishing Status, Tenant Preferred, Bathroom, Point of Contact`. The data is
unprocessed: `Floor` is free text (`"3 out of 5"`, `"Ground out of 2"`), `Posted On`
is a date string, `Area Locality` has 2,235 distinct free-text values, and `Rent`
runs from ₹1,200 to ₹35,00,000 with a skewness of 21.4.

---

## Running it

```bash
pip install -r requirements.txt

python rent_prediction.py
streamlit run app.py
```

Training takes about a minute. `models/` and `outputs/` are committed, so
`streamlit run app.py` works on a fresh clone without the dataset.

> **Use one interpreter for both steps.** Pickled scikit-learn and XGBoost models
> are not portable across library versions. The app records the versions it was
> trained with and stops with a clear message if they do not match. It has been
> checked under Python 3.11 and Anaconda Python 3.13 with scikit-learn 1.7.2 and
> XGBoost 3.2.0.

---

## Pipeline

**Preprocessing**
- Drop duplicate rows (none in this file) and check for nulls (none).
- Parse `Floor` into `Floor_No` and `Total_Floors` (Ground = 0, Upper Basement = −1,
  Lower Basement = −2) and derive `Floor_Ratio = Floor_No / Total_Floors`. Four
  listings give only their own floor; they are treated as the top floor.
- Parse `Posted On` into `Posted_Month`.
- Remove rent outliers with the IQR rule: keep ₹1,000 ≤ Rent ≤ Q3 + 3·IQR
  (₹1,02,000). This removes 280 listings, leaving 4,466.
- Model `y = log1p(Rent)` to tame the right skew: after outlier removal the
  skewness of `Rent` is 1.88 and of `log1p(Rent)` 0.37. Predictions are converted
  back with `expm1` before any rupee metric is computed.

**Feature engineering:** `Size_per_BHK`, `Bath_per_BHK`, `Rooms_Total`, `Log_Size`,
and `Locality_Freq` (how many training listings share the locality).

**Encoding and scaling**, in one scikit-learn `ColumnTransformer` inside each
model's `Pipeline`, so every step is fitted on training data only:
- `StandardScaler` on the 11 numeric features and `Locality_Freq`
- `OneHotEncoder` on Area Type, City, Furnishing Status, Tenant Preferred, Point of Contact
- `OneHotEncoder(min_frequency=10, max_categories=80)` on Area Locality, so rare
  and unseen localities share one "infrequent" column

**Evaluation:** 80/20 train-test split (`random_state=42`), 3-fold cross-validated
R² on the training set, then R² (rupees and log scale), RMSE, MAE and MAPE on the
test set. The model with the best mean CV R² is selected; the test set is not
used to choose it.

### Models

| Model | Settings |
|---|---|
| Linear Regression | baseline |
| Decision Tree | `max_depth=12, min_samples_leaf=5` |
| Random Forest (bagging) | `n_estimators=150, max_depth=18, min_samples_leaf=2` |
| Gradient Boosting | `n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.9` |
| XGBoost | `n_estimators=350, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, tree_method="hist"` |
| Voting Ensemble | LR + RF + GB + XGB, weights 2 : 1 : 2 : 3 |
| Stacking Ensemble | LR + RF + GB + XGB, `RidgeCV` meta-learner, `cv=3` |

The ensembles deliberately mix a linear model with tree models. They make
different kinds of errors, and averaging errors that are not perfectly correlated
cancels part of them.

### Results

Test set: 894 listings. Sorted by 3-fold CV R² on the training set.

| Model | CV R² (log) | Test R² | Test R² (log) | RMSE | MAE | MAPE |
|---|---|---|---|---|---|---|
| **Voting Ensemble** | **0.787 ± 0.003** | 0.770 | 0.796 | ₹9,419 | ₹5,552 | 27.5% |
| Stacking Ensemble | 0.787 ± 0.003 | 0.770 | 0.796 | ₹9,419 | ₹5,541 | 27.5% |
| Gradient Boosting | 0.783 ± 0.001 | 0.753 | 0.787 | ₹9,772 | ₹5,806 | 28.3% |
| XGBoost | 0.772 ± 0.002 | 0.771 | 0.790 | ₹9,404 | ₹5,638 | 28.0% |
| Random Forest | 0.768 ± 0.004 | 0.767 | 0.773 | ₹9,497 | ₹5,743 | 29.1% |
| Linear Regression | 0.761 ± 0.008 | 0.715 | 0.769 | ₹10,486 | ₹6,000 | 29.4% |
| Decision Tree | 0.694 ± 0.013 | 0.725 | 0.717 | ₹10,311 | ₹6,344 | 33.2% |

The voting and stacking ensembles are effectively tied, and XGBoost's slightly
higher test R² is within the noise of a single 894-row test set. The ensembles
are consistently better on log-scale R², MAE, MAPE and CV R², and far more
stable than a single tree. Stacking's learnt weights (LR 0.32, RF 0.13, GB 0.23,
XGB 0.33) are close to the voting weights, which is why the two agree so
closely; linear regression earns a large weight despite being the weakest model
alone because its errors are the least correlated with the tree models'.

The metrics, stacking weights and error correlations are saved in `outputs/results.json`.

---

## What the app shows

Fill in the listing in the sidebar and submit. Changing the city refreshes the
locality list.

- **Fair rent.** The selected model's estimate, with a range of ± its test-set
  MAPE. An asking rent, if entered, is placed on the same scale and judged
  below, within or above the range.
- **This listing, field by field.** XGBoost's exact TreeSHAP contributions for
  this listing, grouped by input field and shown as multipliers on the rent.
- **Model by model.** What all seven models predict for this listing, beside
  their CV R² and test RMSE, MAE and MAPE.
- **What moves the number.** Mean absolute SHAP contribution per input field
  over the test set.

The app says so when the chosen locality has fewer than 10 training listings, and
when an estimate approaches the ₹1,02,000 outlier cut-off, above which the models
have seen no data.

---

## Project structure

```
FairStay/
├── .streamlit/config.toml       app theme
├── data/
│   └── House_Rent_Dataset.csv   (download from Kaggle - not committed)
├── models/                      seven fitted pipelines (.joblib)
├── outputs/
│   ├── figures/                 EDA and result figures
│   ├── app_metadata.json        what the app needs besides the models
│   ├── results.json             metrics, stacking weights, error correlations
│   └── model_comparison.csv     metrics per model
├── rent_features.py             feature code shared by training and the app
├── rent_prediction.py           the whole pipeline: clean, train, evaluate, save
├── app.py                       Streamlit app
├── requirements.txt
└── README.md
```

---

## Limitations

- Asking rents from listings posted April–July 2022, not agreed rents or today's market.
- 1,465 of the 2,235 localities appear only once, and about a third of test
  listings are in a locality the models never saw in training, so location is
  captured coarsely.
- The outlier cut-off removes the top of the market (206 of the 280 removed
  listings are in Mumbai); the models are not valid above ₹1,02,000.
- Converting a log-scale prediction back with `expm1` gives something closer to
  a median than a mean, so the models under-predict expensive listings on average.
- `Point of Contact` is a strong predictor because agents handle pricier flats
  in every city; it marks the market segment rather than causing the rent.

> Predictions are based on historical listing patterns, not a real-time market valuation.
