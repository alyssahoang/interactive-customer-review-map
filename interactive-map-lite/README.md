# Interactive Map Lite

This Streamlit app is the presentation layer for `analysis-v0` outputs.

It loads from repository-root `data/`:

- `data/projection_views/viz_{umap,tsne,pca}.csv` (preferred) or `.parquet`
- `data/projection_views/topic_proxy_lookup.csv` (preferred) or `.parquet`
- `data/projection_views/review_text_lookup.csv`
- `data/projection_views/wordcloud_terms.csv` (preferred) or `.parquet`
- `data/olist_order_reviews_dataset.csv`
- `data/olist_order_reviews_dataset_translated.csv`
- `artifacts/analysis-v1/<run>/semantic_map_points.csv`

## Features kept for project scope

- 2D/3D semantic projection map
- Color-by topic, mood (sentiment), or review group (cluster)
- Optional topic/cluster region-circle overlays
- kNN explorer for local neighborhood interpretation
- Wordcloud and time-trend analytics
- Export filtered table as CSV

## Run locally

From repository root:

```bash
pip install -r interactive-map-lite/requirements.txt
streamlit run interactive-map-lite/app.py
```

## Data volume note

- Full dataset after filtering/preprocessing is `36,567` reviews.
- Projection files are stacked by model (`3 x 36,567 = 109,701` rows total).
- Default map view shows one model at a time.

## Troubleshooting

- If the app starts with missing data warnings, verify files above exist under `data/`.
- If snippets are empty, verify `review_text_lookup.csv` includes `review_id`.
- If PNG download is unavailable, verify `kaleido` is installed in the active environment.
