# ==========================================
# ROBUUSTE CBS WIJKCLUSTERING
# ==========================================
import warnings
warnings.filterwarnings("ignore")

import re
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Configuratie
# ----------------------------
TABLE_ID = "85618NED"
BASE_URL = f"https://opendata.cbs.nl/ODataApi/OData/{TABLE_ID}"
RANDOM_STATE = 42
TIMEOUT = 60
MIN_K = 2
MAX_K = 8

MISSING_TOKENS = {
    ".": np.nan,
    "..": np.nan,
    "x": np.nan,
    "X": np.nan,
    "-": np.nan,
    " ": np.nan,
    "": np.nan,
}

# ----------------------------
# HTTP / OData helpers
# ----------------------------
def build_session() -> Session:
    """Maak een requests-sessie met retries voor tijdelijke netwerkfouten."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def build_url(endpoint: str, params: Optional[dict] = None) -> str:
    """Bouw veilig een URL met queryparameters op."""
    prepared = requests.Request("GET", endpoint, params=params).prepare()
    return prepared.url


def get_odata(url: str, session: Session, timeout: int = TIMEOUT) -> pd.DataFrame:
    """Lees alle pagina's van een OData-endpoint uit en retourneer één DataFrame."""
    rows = []
    next_url = url

    while next_url:
        response = session.get(next_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(f"Onverwachte API-respons voor URL: {next_url}")

        chunk = payload.get("value")
        if chunk is None:
            chunk = payload.get("d", {}).get("results", [])

        if not isinstance(chunk, list):
            raise ValueError(f"Kon geen lijst met records uitlezen voor URL: {next_url}")

        rows.extend(chunk)
        next_url = (
            payload.get("@odata.nextLink")
            or payload.get("odata.nextLink")
            or payload.get("d", {}).get("__next")
        )

    return pd.DataFrame(rows)


# ----------------------------
# Metadata helpers
# ----------------------------
def require_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Ontbrekende kolommen in {df_name}: {missing}")


def prepare_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def find_key(
    metadata: pd.DataFrame,
    include_terms: Optional[Iterable[str]] = None,
    exclude_terms: Optional[Iterable[str]] = None,
    type_contains: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """
    Zoek een DataProperties-key op basis van Title/Description en eventueel Type.
    """
    include_terms = list(include_terms or [])
    exclude_terms = list(exclude_terms or [])

    require_columns(metadata, ["Title", "Description", "Key", "Type"], "metadata")
    meta = metadata.copy()

    search_text = (meta["Title"].fillna("") + " " + meta["Description"].fillna("")).str.lower()
    mask = pd.Series(True, index=meta.index)

    for term in include_terms:
        mask &= search_text.str.contains(re.escape(term.lower()), na=False, regex=True)

    for term in exclude_terms:
        mask &= ~search_text.str.contains(re.escape(term.lower()), na=False, regex=True)

    if type_contains:
        mask &= meta["Type"].str.contains(type_contains, case=False, na=False)

    matches = meta.loc[mask].copy()

    if not matches.empty:
        matches["_title_len"] = matches["Title"].fillna("").str.len()
        return (
            matches.sort_values(by=["_title_len", "Key"], ascending=[True, True])
            .iloc[0]["Key"]
            .strip()
        )

    if required:
        raise KeyError(
            f"Veld niet gevonden. include_terms={include_terms}, "
            f"exclude_terms={exclude_terms}, type_contains={type_contains}"
        )

    return None


def get_feature_title_map(metadata: pd.DataFrame) -> dict:
    require_columns(metadata, ["Key", "Title"], "metadata")
    return (
        metadata[["Key", "Title"]]
        .dropna()
        .assign(Key=lambda x: x["Key"].astype(str).str.strip())
        .drop_duplicates(subset=["Key"])
        .set_index("Key")["Title"]
        .to_dict()
    )


def get_region_name_map(session: Session, region_key: str) -> dict:
    geo_url = build_url(f"{BASE_URL}/{region_key}", {"$format": "json"})
    geo_metadata = get_odata(geo_url, session=session)
    require_columns(geo_metadata, ["Key", "Title"], "geo_metadata")
    return dict(
        zip(
            geo_metadata["Key"].astype(str).str.strip(),
            geo_metadata["Title"].astype(str).str.strip(),
        )
    )


# ----------------------------
# Data helpers
# ----------------------------
def clean_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' ontbreekt in de dataset.")
        df[col] = pd.to_numeric(df[col].replace(MISSING_TOKENS), errors="coerce")
    return df


def choose_best_k(
    X_scaled: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    random_state: int = RANDOM_STATE,
) -> tuple[int, dict]:
    """Bepaal een redelijke k op basis van silhouette score."""
    n_samples = len(X_scaled)
    if n_samples < 3:
        raise ValueError("Te weinig observaties om clustering zinvol uit te voeren.")

    upper_k = min(max_k, n_samples - 1)
    if upper_k < min_k:
        return 2, {}

    scores = {}
    best_score = -1.0
    best_k = min_k

    for k in range(min_k, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_scaled)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(X_scaled, labels)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores


def relabel_clusters_by_income(df: pd.DataFrame, cluster_col: str, income_col: str) -> pd.Series:
    """Maak clusternummers stabiel en logisch: laag inkomen -> hoog inkomen."""
    order = (
        df.groupby(cluster_col)[income_col]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
    return df[cluster_col].map(mapping)


def format_nl_number(value: float, decimals: int = 0, prefix: str = "") -> str:
    if pd.isna(value):
        return f"{prefix}n.v.t."
    text = f"{value:,.{decimals}f}"
    text = text.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefix}{text}"


# ----------------------------
# Hoofdlogica
# ----------------------------
def main() -> None:
    session = build_session()

    print("1/5 Metadata ophalen...")
    metadata_url = build_url(f"{BASE_URL}/DataProperties", {"$format": "json"})
    metadata = get_odata(metadata_url, session=session)
    metadata = prepare_text_columns(metadata, ["Title", "Description", "Key", "Type"])
    require_columns(metadata, ["Title", "Description", "Key", "Type"], "metadata")

    top_meta = metadata[metadata["Type"].str.contains("Topic", case=False, na=False)].copy()
    feature_titles = get_feature_title_map(metadata)

    print("2/5 Relevante variabelen bepalen...")
    income_key = find_key(top_meta, include_terms=["gestandaardiseerd inkomen"], required=True)
    density_key = find_key(top_meta, include_terms=["bevolkingsdichtheid"], required=True)
    woning_key = find_key(top_meta, include_terms=["woningvoorraad"])

    region_code_key = find_key(metadata, type_contains="GeoDetail", required=True)

    chosen_features = {
        "inkomen": income_key,
        "dichtheid": density_key,
    }
    if woning_key:
        chosen_features["woningen"] = woning_key

    print("3/5 Wijknamen ophalen...")
    name_map = get_region_name_map(session=session, region_key=region_code_key)

    print("4/5 Dataset downloaden...")
    feature_keys = list(chosen_features.values())
    selected_keys = list(dict.fromkeys([region_code_key] + feature_keys))

    data_url = build_url(
        f"{BASE_URL}/TypedDataSet",
        {
            "$format": "json",
            "$select": ",".join(selected_keys),
            "$filter": f"startswith({region_code_key},'WK')",
        },
    )
    cbs_df = get_odata(data_url, session=session)

    if cbs_df.empty:
        raise ValueError("Geen wijkrecords opgehaald uit TypedDataSet.")

    require_columns(cbs_df, selected_keys, "cbs_df")

    print("5/5 Data opschonen en clusteren...")
    cbs_df = clean_numeric_columns(cbs_df, feature_keys)

    # Alleen rijen verwijderen waar alle gekozen features ontbreken.
    cbs_df = cbs_df.dropna(subset=feature_keys, how="all").copy()
    cbs_df[region_code_key] = cbs_df[region_code_key].astype(str).str.strip()

    if len(cbs_df) < 3:
        raise ValueError("Te weinig bruikbare wijkrecords over na opschoning.")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_scaled = pipeline.fit_transform(cbs_df[feature_keys])

    best_k, silhouette_scores = choose_best_k(X_scaled)
    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    cbs_df["cluster_raw"] = model.fit_predict(X_scaled)
    cbs_df["cluster"] = relabel_clusters_by_income(cbs_df, "cluster_raw", income_key).astype(int)

    # ----------------------------
    # Visualisatie
    # ----------------------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_comp = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cluster_id in sorted(cbs_df["cluster"].unique()):
        mask = cbs_df["cluster"] == cluster_id
        axes[0].scatter(
            pca_comp[mask, 0],
            pca_comp[mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
        )

    axes[0].set_title(f"Wijkclustering (PCA, k={best_k})")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend()

    scatter = axes[1].scatter(
        cbs_df[income_key],
        cbs_df[density_key],
        c=cbs_df["cluster"],
        cmap="viridis",
        alpha=0.6,
    )
    axes[1].set_title("Inkomen vs. bevolkingsdichtheid")
    axes[1].set_xlabel(feature_titles.get(income_key, income_key))
    axes[1].set_ylabel(feature_titles.get(density_key, density_key))
    fig.colorbar(scatter, ax=axes[1], label="Cluster")

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Conclusie / clusterprofielen
    # ----------------------------
    print("\nSilhouette-scores per k:")
    if silhouette_scores:
        for k, score in silhouette_scores.items():
            print(f"  k={k}: {score:.3f}")
    else:
        print("  Geen silhouette-scores beschikbaar; default clustering gebruikt.")

    print("\nClusterprofielen:")
    cluster_means = cbs_df.groupby("cluster")[feature_keys].mean().sort_index()

    for cluster_id, row in cluster_means.iterrows():
        cluster_rows = cbs_df[cbs_df["cluster"] == cluster_id].copy()
        example_codes = cluster_rows[region_code_key].head(3).tolist()
        example_names = [name_map.get(code, code) for code in example_codes]

        parts = [
            f"gem. inkomen {format_nl_number(row[income_key] * 1000, 0, prefix='€')}",
            f"gem. dichtheid {format_nl_number(row[density_key], 0)}",
        ]

        if woning_key:
            parts.append(f"gem. woningvoorraad {format_nl_number(row[woning_key], 0)}")

        print(
            f"- Cluster {cluster_id}: "
            + "; ".join(parts)
            + f". Voorbeelden: {', '.join(example_names)}"
        )


if __name__ == "__main__":
    main()