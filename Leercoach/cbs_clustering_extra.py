# -*- coding: utf-8 -*-
"""
CBS wijkclustering + geo-informatie + interactieve kaarten.

Wat dit script doet:
1. Haalt CBS StatLine-data op uit tabel 85618NED.
2. Detecteert relevante variabelen dynamisch via DataProperties.
3. Voert clustering uit op wijkniveau.
4. Haalt wijkgeometrieën op via de PDOK/CBS OGC API.
5. Koppelt statistiek aan geometrie via wijkcode (WK...).
6. Exporteert GeoJSON.
7. Maakt optioneel een interactieve Folium-kaart.
8. Maakt optioneel een losse MapLibre HTML-pagina.

Benodigde packages:
    pip install pandas numpy requests scikit-learn matplotlib geopandas shapely folium
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

try:
    import folium
    from folium.features import GeoJsonTooltip

    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


# =========================================================
# CONFIGURATIE
# =========================================================
TABLE_ID = "85618NED"
BASE_URL = f"https://opendata.cbs.nl/ODataApi/OData/{TABLE_ID}"
RANDOM_STATE = 42
TIMEOUT = 60

CBS_YEAR = 2023
PDOK_YEAR = 2023

MIN_K = 2
MAX_K = 8

# Zet op bijvoorbeeld "0363" voor Amsterdam om sneller te testen.
GEMEENTE_CODE_FILTER: Optional[str] = None

# Ondersteund: "wijk", "buurt", "gemeente"
GEO_LEVEL = "wijk"

OUTPUT_DIR = Path(".")
OUTPUT_GEOJSON = OUTPUT_DIR / f"{GEO_LEVEL}en_clusters_{PDOK_YEAR}.geojson"
OUTPUT_FOLIUM_HTML = OUTPUT_DIR / f"{GEO_LEVEL}en_clusters_{PDOK_YEAR}_folium.html"
OUTPUT_MAPLIBRE_HTML = OUTPUT_DIR / f"{GEO_LEVEL}en_clusters_{PDOK_YEAR}_maplibre.html"

MISSING_TOKENS = {
    ".": np.nan,
    "..": np.nan,
    "x": np.nan,
    "X": np.nan,
    "-": np.nan,
    " ": np.nan,
    "": np.nan,
}

CLUSTER_COLORS = [
    "#440154", "#3b528b", "#21918c", "#5ec962",
    "#fde725", "#ff7f0e", "#d62728", "#9467bd",
]


# =========================================================
# HTTP HELPERS
# =========================================================
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
    prepared = requests.Request("GET", endpoint, params=params).prepare()
    return prepared.url


def get_odata(url: str, session: Session, timeout: int = TIMEOUT) -> pd.DataFrame:
    """Lees alle pagina's van een OData-endpoint uit en geef één DataFrame terug."""
    rows: list[dict] = []
    next_url = url

    while next_url:
        response = session.get(next_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(f"Onverwachte API-respons voor {next_url}")

        chunk = payload.get("value")
        if chunk is None:
            chunk = payload.get("d", {}).get("results", [])

        if not isinstance(chunk, list):
            raise ValueError(f"Kon records niet uitlezen voor {next_url}")

        rows.extend(chunk)
        next_url = (
            payload.get("@odata.nextLink")
            or payload.get("odata.nextLink")
            or payload.get("d", {}).get("__next")
        )

    return pd.DataFrame(rows)


# =========================================================
# ALGEMENE HELPERS
# =========================================================
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
    """Zoek een DataProperties-key op basis van Title/Description/Type."""
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
        mask &= meta["Type"].astype(str).str.contains(type_contains, case=False, na=False)

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


def format_nl_number(value: float, decimals: int = 0, prefix: str = "") -> str:
    if pd.isna(value):
        return f"{prefix}n.v.t."
    text = f"{value:,.{decimals}f}"
    text = text.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefix}{text}"


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
) -> tuple[int, dict[int, float]]:
    """Kies een redelijke k op basis van silhouette score."""
    n_samples = len(X_scaled)
    if n_samples < 3:
        raise ValueError("Te weinig observaties om clustering uit te voeren.")

    upper_k = min(max_k, n_samples - 1)
    if upper_k < min_k:
        return 2, {}

    scores: dict[int, float] = {}
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
    """Maak clusternummers stabiel: laag inkomen -> hoog inkomen."""
    order = (
        df.groupby(cluster_col)[income_col]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
    return df[cluster_col].map(mapping)


# =========================================================
# CBS-DATA
# =========================================================
def load_cbs_data(session: Session) -> tuple[pd.DataFrame, dict, dict]:
    print("1/7 Metadata ophalen...")
    metadata_url = build_url(f"{BASE_URL}/DataProperties", {"$format": "json"})
    metadata = get_odata(metadata_url, session=session)
    metadata = prepare_text_columns(metadata, ["Title", "Description", "Key", "Type"])
    require_columns(metadata, ["Title", "Description", "Key", "Type"], "metadata")

    top_meta = metadata[metadata["Type"].astype(str).str.contains("Topic", case=False, na=False)].copy()
    feature_titles = get_feature_title_map(metadata)

    print("2/7 CBS-variabelen bepalen...")
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

    print("3/7 CBS-data downloaden...")
    feature_keys = list(chosen_features.values())
    selected_keys = list(dict.fromkeys([region_code_key] + feature_keys))

    # CBS wijkcodes beginnen met WK
    filter_expr = f"startswith({region_code_key},'WK')"
    if GEMEENTE_CODE_FILTER:
        gm = str(GEMEENTE_CODE_FILTER).zfill(4)
        filter_expr = f"startswith({region_code_key},'WK{gm}')"

    data_url = build_url(
        f"{BASE_URL}/TypedDataSet",
        {
            "$format": "json",
            "$select": ",".join(selected_keys),
            "$filter": filter_expr,
        },
    )
    cbs_df = get_odata(data_url, session=session)

    if cbs_df.empty:
        raise ValueError("Geen wijkrecords opgehaald uit TypedDataSet.")

    require_columns(cbs_df, selected_keys, "cbs_df")
    cbs_df = clean_numeric_columns(cbs_df, feature_keys)
    cbs_df[region_code_key] = cbs_df[region_code_key].astype(str).str.strip()

    # Alleen verwijderen als alle features ontbreken.
    cbs_df = cbs_df.dropna(subset=feature_keys, how="all").copy()

    return cbs_df, chosen_features, {
        "feature_titles": feature_titles,
        "region_code_key": region_code_key,
    }


# =========================================================
# CLUSTERING
# =========================================================
def cluster_cbs_data(cbs_df: pd.DataFrame, chosen_features: dict) -> tuple[pd.DataFrame, dict]:
    print("4/7 Clustering uitvoeren...")
    feature_keys = list(chosen_features.values())

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
    cbs_df["cluster"] = relabel_clusters_by_income(
        cbs_df,
        cluster_col="cluster_raw",
        income_col=chosen_features["inkomen"],
    ).astype(int)

    return cbs_df, {
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
        "X_scaled": X_scaled,
    }


# =========================================================
# PDOK OGC API HELPERS
# =========================================================
def get_pdok_ogc_base_url(year: int) -> str:
    return f"https://api.pdok.nl/cbs/wijken-en-buurten-{year}/ogc/v1"


def get_pdok_collection_name(level: str = "wijk") -> str:
    level = level.lower().strip()
    if level == "wijk":
        return "wijken"
    if level == "buurt":
        return "buurten"
    if level == "gemeente":
        return "gemeenten"
    raise ValueError(f"Onbekend geo-niveau: {level}")


def get_next_link(payload: dict) -> Optional[str]:
    for link in payload.get("links", []):
        if link.get("rel") == "next" and link.get("href"):
            return link["href"]
    return None


def detect_region_code_column(df: pd.DataFrame, level: str = "wijk") -> str:
    """Zoek automatisch de kolom met WK-/BU-/GM-codes in de geodata."""
    level = level.lower().strip()
    preferred_candidates: list[str] = []

    for col in df.columns:
        low = col.lower()

        if level == "wijk":
            if (
                low == "wijkcode"
                or "wijkcode" in low
                or low == "statcode"
                or low == "regio_code"
                or low.endswith("_wk")
                or low == "wk_code"
                or low == "jrstatcode"
            ):
                preferred_candidates.append(col)

        elif level == "buurt":
            if (
                low == "buurtcode"
                or "buurtcode" in low
                or low == "statcode"
                or low == "regio_code"
                or low.endswith("_bu")
                or low == "bu_code"
                or low == "jrstatcode"
            ):
                preferred_candidates.append(col)

        elif level == "gemeente":
            if (
                low == "gemeentecode"
                or "gemeentecode" in low
                or low == "statcode"
                or low == "regio_code"
                or low == "jrstatcode"
            ):
                preferred_candidates.append(col)

    for col in preferred_candidates:
        values = df[col].dropna().astype(str).str.strip()
        if values.empty:
            continue

        if level == "wijk" and (values.str.startswith("WK").mean() > 0.5):
            return col
        if level == "buurt" and (values.str.startswith("BU").mean() > 0.5):
            return col
        if level == "gemeente" and (values.str.startswith("GM").mean() > 0.5):
            return col

    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip()
        if values.empty:
            continue

        if level == "wijk" and (values.str.startswith("WK").mean() > 0.5):
            return col
        if level == "buurt" and (values.str.startswith("BU").mean() > 0.5):
            return col
        if level == "gemeente" and (values.str.startswith("GM").mean() > 0.5):
            return col

    raise KeyError(f"Geen regio-codekolom gevonden in geodata. Kolommen: {list(df.columns)}")


def load_pdok_geometries(session: Session, year: int = PDOK_YEAR, level: str = GEO_LEVEL) -> gpd.GeoDataFrame:
    """Haal geometrieën op via de PDOK/CBS OGC API in GeoJSON-formaat."""
    print("5/7 Geo-laag ophalen via PDOK OGC API...")

    collection = get_pdok_collection_name(level)
    base_items_url = f"{get_pdok_ogc_base_url(year)}/collections/{collection}/items"

    params = {
        "f": "json",
        "limit": 1000,
    }

    all_features: list[dict] = []
    next_url: Optional[str] = base_items_url
    is_first_request = True

    while next_url:
        if is_first_request:
            response = session.get(next_url, params=params, timeout=TIMEOUT * 2)
            is_first_request = False
        else:
            response = session.get(next_url, timeout=TIMEOUT * 2)

        response.raise_for_status()
        payload = response.json()

        features = payload.get("features", [])
        if not isinstance(features, list):
            raise ValueError("OGC API gaf geen geldige 'features'-lijst terug.")

        all_features.extend(features)
        next_url = get_next_link(payload)

    if not all_features:
        raise ValueError("Geen geometrieën opgehaald uit de PDOK OGC API.")

    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    if gdf.empty:
        raise ValueError("Lege GeoDataFrame uit PDOK OGC API.")

    code_col = detect_region_code_column(gdf, level=level)
    gdf[code_col] = gdf[code_col].astype(str).str.strip()

    if GEMEENTE_CODE_FILTER:
        gm = str(GEMEENTE_CODE_FILTER).zfill(4)
        if level == "wijk":
            prefix = f"WK{gm}"
        elif level == "buurt":
            prefix = f"BU{gm}"
        else:
            prefix = f"GM{gm}"
        gdf = gdf[gdf[code_col].str.startswith(prefix)].copy()

    if gdf.empty:
        raise ValueError("Geen geometrieën over na toepassing van de gemeentefilter.")

    preferred_name_cols = [c for c in gdf.columns if "naam" in c.lower()]
    keep_cols = [code_col] + preferred_name_cols + ["geometry"]
    keep_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[keep_cols].copy()
    gdf = gdf.rename(columns={code_col: "regio_code"})

    # Vereenvoudig geometrie voor performance.
    gdf["geometry"] = gdf["geometry"].simplify(0.00015, preserve_topology=True)

    return gdf


# =========================================================
# MERGE + EXPORT
# =========================================================
def merge_stats_and_geo(
    geo_gdf: gpd.GeoDataFrame,
    cbs_df: pd.DataFrame,
    region_code_key: str,
    chosen_features: dict,
) -> gpd.GeoDataFrame:
    print("6/7 Statistiek en geometrie koppelen...")

    feature_keys = list(chosen_features.values())
    merge_cols = [region_code_key, "cluster"] + feature_keys

    merged = geo_gdf.merge(
        cbs_df[merge_cols].copy(),
        left_on="regio_code",
        right_on=region_code_key,
        how="left",
    )

    rename_map = {
        chosen_features["inkomen"]: "gem_inkomen_x1000_euro",
        chosen_features["dichtheid"]: "bevolkingsdichtheid",
    }
    if "woningen" in chosen_features:
        rename_map[chosen_features["woningen"]] = "woningvoorraad"

    merged = merged.rename(columns=rename_map)
    merged = merged.dropna(subset=["cluster"]).copy()

    if merged.empty:
        raise ValueError(
            "Na koppeling tussen CBS-data en geodata zijn geen records over. "
            "Controleer of CBS- en PDOK-codes overeenkomen."
        )

    merged["cluster"] = merged["cluster"].astype(int)
    merged["cluster_label"] = merged["cluster"].apply(lambda x: f"Cluster {x}")

    for col in ["gem_inkomen_x1000_euro", "bevolkingsdichtheid", "woningvoorraad"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


def export_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    print(f"7/7 GeoJSON exporteren naar {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")


# =========================================================
# VISUALISATIE
# =========================================================
def plot_static_scatter(
    cbs_df: pd.DataFrame,
    chosen_features: dict,
    feature_titles: dict,
    X_scaled: np.ndarray,
    best_k: int,
) -> None:
    income_key = chosen_features["inkomen"]
    density_key = chosen_features["dichtheid"]

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


# =========================================================
# FOLIUM
# =========================================================
def create_folium_map(gdf: gpd.GeoDataFrame, output_html: Path) -> None:
    if not FOLIUM_AVAILABLE:
        print("Folium is niet geïnstalleerd. Sla deze stap over.")
        return

    if gdf.empty:
        print("Geen geodata beschikbaar voor Folium.")
        return

    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    m = folium.Map(
        location=[center_y, center_x],
        zoom_start=9 if GEMEENTE_CODE_FILTER else 7,
        tiles="CartoDB positron",
        control_scale=True,
    )

    def style_function(feature):
        cluster = feature["properties"].get("cluster")
        color = "#bdbdbd"
        if cluster is not None:
            color = CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]
        return {
            "fillColor": color,
            "color": "#333333",
            "weight": 0.7,
            "fillOpacity": 0.65,
        }

    tooltip_fields: list[str] = []
    tooltip_aliases: list[str] = []

    preferred_pairs = [
        ("regio_code", "Regiocode"),
        ("cluster_label", "Cluster"),
        ("gem_inkomen_x1000_euro", "Gem. inkomen (x1000 euro)"),
        ("bevolkingsdichtheid", "Bevolkingsdichtheid"),
        ("woningvoorraad", "Woningvoorraad"),
    ]
    for col, label in preferred_pairs:
        if col in gdf.columns:
            tooltip_fields.append(col)
            tooltip_aliases.append(label)

    name_cols = [c for c in gdf.columns if "naam" in c.lower()]
    for col in name_cols[:2]:
        if col not in tooltip_fields:
            tooltip_fields.insert(1, col)
            tooltip_aliases.insert(1, col)

    folium.GeoJson(
        gdf.to_json(),
        name="Clusters",
        style_function=style_function,
        tooltip=GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
            labels=True,
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    print(f"Interactieve Folium-kaart opgeslagen als: {output_html}")


# =========================================================
# MAPLIBRE HTML
# =========================================================
def create_maplibre_html(geojson_filename: str, output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang=\"nl\">
<head>
  <meta charset=\"utf-8\" />
  <title>CBS Wijkclusters</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link rel=\"stylesheet\" href=\"https://unpkg.com/maplibre-gl@5.20.2/dist/maplibre-gl.css\" />
  <script src=\"https://unpkg.com/maplibre-gl@5.20.2/dist/maplibre-gl.js\"></script>
  <style>
    html, body, #map {{ margin: 0; padding: 0; height: 100%; width: 100%; }}
    .legend {{
      position: absolute; right: 12px; bottom: 12px; background: white; padding: 10px 12px;
      border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15); font-family: Arial, sans-serif;
      font-size: 13px; z-index: 10;
    }}
    .legend-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
    .legend-color {{ width: 14px; height: 14px; margin-right: 8px; border-radius: 3px; border: 1px solid #666; }}
  </style>
</head>
<body>
<div id=\"map\"></div>
<div class=\"legend\">
  <strong>Clusters</strong>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#440154\"></span>Cluster 0</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#3b528b\"></span>Cluster 1</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#21918c\"></span>Cluster 2</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#5ec962\"></span>Cluster 3</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#fde725\"></span>Cluster 4</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#ff7f0e\"></span>Cluster 5</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#d62728\"></span>Cluster 6</div>
  <div class=\"legend-row\"><span class=\"legend-color\" style=\"background:#9467bd\"></span>Cluster 7</div>
</div>
<script>
  const map = new maplibregl.Map({{
    container: 'map',
    style: {{
      version: 8,
      sources: {{
        osm: {{
          type: 'raster',
          tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
          tileSize: 256,
          attribution: '© OpenStreetMap contributors'
        }}
      }},
      layers: [{{ id: 'osm', type: 'raster', source: 'osm' }}]
    }},
    center: [5.3, 52.1],
    zoom: 7
  }});

  map.addControl(new maplibregl.NavigationControl(), 'top-right');

  map.on('load', async () => {{
    const response = await fetch('./{geojson_filename}');
    const geojson = await response.json();

    map.addSource('wijken', {{ type: 'geojson', data: geojson }});

    map.addLayer({{
      id: 'wijk-fill',
      type: 'fill',
      source: 'wijken',
      paint: {{
        'fill-color': [
          'step', ['to-number', ['coalesce', ['get', 'cluster'], -1]],
          '#bdbdbd',
          0, '#440154',
          1, '#3b528b',
          2, '#21918c',
          3, '#5ec962',
          4, '#fde725',
          5, '#ff7f0e',
          6, '#d62728',
          7, '#9467bd'
        ],
        'fill-opacity': 0.65
      }}
    }});

    map.addLayer({{
      id: 'wijk-outline',
      type: 'line',
      source: 'wijken',
      paint: {{ 'line-color': '#333333', 'line-width': 0.7 }}
    }});

    const bounds = new maplibregl.LngLatBounds();
    for (const feature of geojson.features) {{
      const geom = feature.geometry;
      if (!geom) continue;
      const coords = geom.type === 'Polygon'
        ? geom.coordinates.flat(1)
        : geom.type === 'MultiPolygon'
          ? geom.coordinates.flat(2)
          : [];
      coords.forEach(c => bounds.extend(c));
    }}
    if (!bounds.isEmpty()) {{
      map.fitBounds(bounds, {{ padding: 20 }});
    }}

    const popup = new maplibregl.Popup({{ closeButton: false, closeOnClick: false }});
    map.on('mousemove', 'wijk-fill', (e) => {{
      map.getCanvas().style.cursor = 'pointer';
      const props = e.features[0].properties;
      const html = `
        <strong>${{props.wijknaam || props.naam || props.regio_code || 'Onbekend'}}</strong><br>
        Regio: ${{props.regio_code || '-'}}<br>
        Cluster: ${{props.cluster_label || props.cluster || '-'}}<br>
        Inkomen: ${{props.gem_inkomen_x1000_euro ?? '-'}} (x1000 euro)<br>
        Dichtheid: ${{props.bevolkingsdichtheid ?? '-'}}<br>
        Woningen: ${{props.woningvoorraad ?? '-'}}
      `;
      popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    }});
    map.on('mouseleave', 'wijk-fill', () => {{
      map.getCanvas().style.cursor = '';
      popup.remove();
    }});
  }});
</script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    print(f"MapLibre HTML opgeslagen als: {output_html}")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    session = build_session()

    cbs_df, chosen_features, meta = load_cbs_data(session=session)
    cbs_df, cluster_info = cluster_cbs_data(cbs_df, chosen_features)

    geo_gdf = load_pdok_geometries(session=session, year=PDOK_YEAR, level=GEO_LEVEL)
    merged_gdf = merge_stats_and_geo(
        geo_gdf=geo_gdf,
        cbs_df=cbs_df,
        region_code_key=meta["region_code_key"],
        chosen_features=chosen_features,
    )

    export_geojson(merged_gdf, OUTPUT_GEOJSON)
    create_folium_map(merged_gdf, OUTPUT_FOLIUM_HTML)
    create_maplibre_html(OUTPUT_GEOJSON.name, OUTPUT_MAPLIBRE_HTML)

    print("\nSilhouette-scores per k:")
    if cluster_info["silhouette_scores"]:
        for k, score in cluster_info["silhouette_scores"].items():
            print(f"  k={k}: {score:.3f}")
    else:
        print("  Geen silhouette-scores beschikbaar.")

    print("\nClusterprofielen:")
    profile_cols: list[str] = []
    if "gem_inkomen_x1000_euro" in merged_gdf.columns:
        profile_cols.append("gem_inkomen_x1000_euro")
    if "bevolkingsdichtheid" in merged_gdf.columns:
        profile_cols.append("bevolkingsdichtheid")
    if "woningvoorraad" in merged_gdf.columns:
        profile_cols.append("woningvoorraad")

    cluster_means = merged_gdf.groupby("cluster")[profile_cols].mean().sort_index()
    name_cols = [c for c in merged_gdf.columns if "naam" in c.lower()]
    name_col = name_cols[0] if name_cols else None

    for cluster_id, row in cluster_means.iterrows():
        sub = merged_gdf[merged_gdf["cluster"] == cluster_id]
        if name_col and name_col in sub.columns:
            examples = sub[name_col].dropna().astype(str).head(3).tolist()
        else:
            examples = sub["regio_code"].astype(str).head(3).tolist()

        parts: list[str] = []
        if "gem_inkomen_x1000_euro" in row.index:
            parts.append(
                f"gem. inkomen {format_nl_number(row['gem_inkomen_x1000_euro'] * 1000, 0, prefix='€')}"
            )
        if "bevolkingsdichtheid" in row.index:
            parts.append(
                f"gem. dichtheid {format_nl_number(row['bevolkingsdichtheid'], 0)}"
            )
        if "woningvoorraad" in row.index:
            parts.append(
                f"gem. woningvoorraad {format_nl_number(row['woningvoorraad'], 0)}"
            )

        print(f"- Cluster {cluster_id}: " + "; ".join(parts) + f". Voorbeelden: {', '.join(examples)}")

    plot_static_scatter(
        cbs_df=cbs_df,
        chosen_features=chosen_features,
        feature_titles=meta["feature_titles"],
        X_scaled=cluster_info["X_scaled"],
        best_k=cluster_info["best_k"],
    )

    print(f"\nKlaar. GeoJSON: {OUTPUT_GEOJSON}")
    if FOLIUM_AVAILABLE:
        print(f"Klaar. Folium HTML: {OUTPUT_FOLIUM_HTML}")
    print(f"Klaar. MapLibre HTML: {OUTPUT_MAPLIBRE_HTML}")


if __name__ == "__main__":
    main()
