# Interactive Map Lite

This Streamlit app is the presentation layer for `analysis-v0` outputs.

It loads only from the submission package:

- `submission/data/phase3_multi/projection_views/viz_{umap,tsne,pca}.parquet`
- `submission/data/phase3_multi/projection_views/topic_proxy_lookup.parquet`
- `submission/data/phase3_multi/projection_views/review_text_lookup.csv`
- `submission/data/phase3_multi/projection_views/wordcloud_terms.parquet`
- `submission/data/olist_order_reviews_dataset.csv`
- `submission/data/olist_order_reviews_dataset_translated.csv`

## Features kept for project scope

- 2D/3D semantic projection map
- Color-by topic, mood (sentiment), or review group (cluster)
- Optional topic/cluster region-circle overlays
- kNN explorer for local neighborhood interpretation
- Wordcloud and time-trend analytics
- Export filtered table as CSV

## Run locally

From `submission/interactive-map-lite`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or from `submission` root:

```bash
pip install -r interactive-map-lite/requirements.txt
streamlit run interactive-map-lite/app.py
```

## Data volume note

- Each projection file contains `109,701` rows total (3 models x 36,567 reviews).
- App default is one model (`MiniLM`), so map view starts around `36,567` points.
- If multiple models are selected, map rows increase accordingly.

## Troubleshooting

- If the app starts with missing data warnings, verify all files above exist under `submission/data`.
- If snippets are empty, verify `review_text_lookup.csv` is present and includes `review_id`.
- If PNG download is unavailable, verify `kaleido` is installed in the active environment.
