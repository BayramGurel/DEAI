
# %% [markdown]
# # Datapunt 1 - Predictive Modeling met RDW Open Data
#
# Doel: voorspel de `catalogusprijs` van voertuigen met een baseline,
# Linear Regression en Decision Tree Regressor.
#
# Benodigde packages:
# pip install pandas numpy requests scikit-learn matplotlib

# %%
import warnings
warnings.filterwarnings("ignore")

import math
from datetime import datetime
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
BASE_URL = "https://opendata.rdw.nl/resource/m9d7-ebf2.json"


def fetch_json(url, params=None, timeout=60):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def preview_columns():
    preview = fetch_json(BASE_URL, params={"$limit": 5})
    if not preview:
        raise RuntimeError("RDW API gaf geen preview-data terug.")
    return pd.DataFrame(preview).columns.tolist()


def first_available(available_columns, candidates, required=False, label="kolom"):
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    if required:
        raise KeyError(f"Geen bruikbare {label} gevonden. Geprobeerd: {candidates}")
    return None


def fetch_rdw_dataset(target_rows=5000, batch_size=2000, max_raw_rows=20000):
    available = preview_columns()

    target_col = first_available(available, ["catalogusprijs"], required=True, label="doelkolom")
    weight_col = first_available(
        available,
        ["massa_ledig_voertuig", "massa_rijklaar", "toegestane_maximum_massa_voertuig"],
        required=True,
        label="gewichtskolom",
    )
    date_col = first_available(
        available,
        ["datum_eerste_toelating_dt", "datum_eerste_toelating"],
        required=True,
        label="datumkolom",
    )

    optional_numeric = [
        "cilinderinhoud",
        "aantal_cilinders",
        "aantal_deuren",
        "aantal_zitplaatsen",
        "vermogen_massarijklaar",
        "laadvermogen",
    ]
    optional_categorical = [
        "merk",
        "inrichting",
        "voertuigsoort",
        "brandstof_omschrijving",
        "eerste_kleur",
    ]

    selected_columns = [target_col, weight_col, date_col]
    selected_columns += [c for c in optional_numeric if c in available]
    selected_columns += [c for c in optional_categorical if c in available]
    selected_columns = list(dict.fromkeys(selected_columns))

    where_parts = [
        f"{target_col} IS NOT NULL",
        f"{weight_col} IS NOT NULL",
        f"{date_col} IS NOT NULL",
    ]
    if "voertuigsoort" in available:
        where_parts.append("voertuigsoort = 'Personenauto'")

    frames = []
    offset = 0
    total_downloaded = 0

    while total_downloaded < max_raw_rows:
        params = {
            "$select": ",".join(selected_columns),
            "$where": " AND ".join(where_parts),
            "$limit": batch_size,
            "$offset": offset,
        }
        rows = fetch_json(BASE_URL, params=params)
        if not rows:
            break
        chunk = pd.DataFrame(rows)
        frames.append(chunk)
        total_downloaded += len(chunk)
        offset += batch_size
        if len(chunk) < batch_size or total_downloaded >= target_rows:
            break

    if not frames:
        raise RuntimeError("Geen RDW-data opgehaald.")

    df = pd.concat(frames, ignore_index=True)
    return df, target_col, weight_col, date_col


rdw_df_raw, target_col, weight_col, date_col = fetch_rdw_dataset(target_rows=6000)
print("Ruwe dataset-vorm:", rdw_df_raw.shape)
print("Kolommen:", rdw_df_raw.columns.tolist())
rdw_df_raw.head()

# %%
rdw_df = rdw_df_raw.copy()

for col in rdw_df.columns:
    if col == date_col:
        continue
    rdw_df[col] = pd.to_numeric(rdw_df[col], errors="ignore")

rdw_df[date_col] = pd.to_datetime(rdw_df[date_col], errors="coerce")
rdw_df["bouwjaar"] = rdw_df[date_col].dt.year
current_year = datetime.now().year

numeric_candidates = [
    target_col,
    weight_col,
    "bouwjaar",
    "cilinderinhoud",
    "aantal_cilinders",
    "aantal_deuren",
    "aantal_zitplaatsen",
    "vermogen_massarijklaar",
    "laadvermogen",
]

for col in numeric_candidates:
    if col in rdw_df.columns:
        rdw_df[col] = pd.to_numeric(rdw_df[col], errors="coerce")

rdw_df = rdw_df[(rdw_df[target_col] > 0) & (rdw_df["bouwjaar"].between(1950, current_year))]
rdw_df = rdw_df.dropna(subset=[target_col, weight_col, "bouwjaar"])

# Optionele features alleen behouden als ze genoeg informatie bevatten
feature_pool = [c for c in rdw_df.columns if c not in [target_col, date_col]]
valid_features = []
for col in feature_pool:
    non_null_ratio = rdw_df[col].notna().mean()
    n_unique = rdw_df[col].nunique(dropna=True)
    if non_null_ratio >= 0.20 and n_unique > 1:
        valid_features.append(col)

if len(rdw_df) < 1000:
    raise RuntimeError(f"Na opschoning zijn minder dan 1000 rijen over: {len(rdw_df)}")

print("Opgeschoonde dataset-vorm:", rdw_df.shape)
print("Aantal bruikbare rijen:", len(rdw_df))
print("Features:", valid_features)
rdw_df[valid_features + [target_col]].head()

# %%
X = rdw_df[valid_features].copy()
y = rdw_df[target_col].copy()

numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

numeric_transformer_linear = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_transformer_tree = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_linear, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_tree, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

models = {
    "Baseline (gemiddelde)": DummyRegressor(strategy="mean"),
    "Linear Regression": Pipeline(
        steps=[("preprocess", preprocessor_linear), ("model", LinearRegression())]
    ),
    "Decision Tree": Pipeline(
        steps=[
            ("preprocess", preprocessor_tree),
            (
                "model",
                DecisionTreeRegressor(
                    random_state=RANDOM_STATE,
                    max_depth=8,
                    min_samples_leaf=20,
                ),
            ),
        ]
    ),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    results.append(
        {
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2": r2_score(y_test, y_pred),
        }
    )

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
results_df

# %%
best_model_name = results_df.iloc[0]["Model"]
print("Beste model op basis van laagste RMSE:", best_model_name)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions[best_model_name], alpha=0.5)
min_val = min(y_test.min(), predictions[best_model_name].min())
max_val = max(y_test.max(), predictions[best_model_name].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Werkelijke catalogusprijs")
plt.ylabel("Voorspelde catalogusprijs")
plt.title(f"Werkelijk vs voorspeld - {best_model_name}")
plt.tight_layout()
plt.show()

# %%
best_row = results_df.iloc[0]
worst_row = results_df.iloc[-1]

conclusion = f"""
Technische conclusie
--------------------
De dataset bevat {len(rdw_df):,} bruikbare voertuigen na opschoning.
Het beste model is: {best_row['Model']}.

Belangrijkste metrics van het beste model:
- MAE: {best_row['MAE']:.2f}
- RMSE: {best_row['RMSE']:.2f}
- R²: {best_row['R2']:.3f}

Interpretatie:
- Hoe lager de MAE en RMSE, hoe beter de voorspellingen gemiddeld aansluiten op de echte catalogusprijs.
- Hoe hoger de R², hoe beter het model variantie in de prijs verklaart.
- Vergeleken met de baseline presteert {best_row['Model']} beter dan alleen het gemiddelde voorspellen.

Aandachtspunt:
- Deze uitkomst hangt af van de kwaliteit van de beschikbare RDW-kenmerken.
- Extra feature engineering is bewust buiten scope gehouden om het semesterdoel compact te houden.
"""
print(conclusion)
