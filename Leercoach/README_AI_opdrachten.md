
# AI-opdrachten Bayram Gurel

Deze map bevat vier bestanden:

- `rdw_predictive_model.py`  -> script voor supervised learning (RDW)
- `cbs_clustering.py`        -> script voor unsupervised learning (CBS)
- `rdw_predictive_model.ipynb`
- `cbs_clustering.ipynb`

## Installatie
Gebruik bij voorkeur een virtuele omgeving en installeer daarna:

```bash
pip install pandas numpy requests scikit-learn matplotlib notebook
```

## Starten als script
```bash
python rdw_predictive_model.py
python cbs_clustering.py
```

## Starten als notebook
```bash
jupyter notebook
```
Open daarna het juiste `.ipynb` bestand.

## Wat elk bestand doet
### RDW notebook/script
- haalt RDW Open Data op
- maakt een baseline-model
- traint Linear Regression
- traint een Decision Tree Regressor
- vergelijkt modellen met MAE, RMSE en R²
- print een technische conclusie

### CBS notebook/script
- haalt CBS Kerncijfers Wijken en Buurten op
- zoekt metadata op om bruikbare kolommen te vinden
- voert de elbow-methode uit
- traint K-Means
- maakt 2 cluster-visualisaties
- toont clusterprofielen en een conclusie
