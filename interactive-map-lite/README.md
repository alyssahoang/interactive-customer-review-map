# Interactive Customer Review Map (Olist)

This app helps users explore how customer reviews are organized in semantic space, with a focus on sentiment and service-topic patterns in Olist reviews.

It is designed for non-technical and technical audiences:
- non-technical users can quickly identify complaint hotspots, topic patterns, and example reviews
- technical users can inspect projection choices and clustering diagnostics

## What This App Shows

- **Each dot = one review**
- Dot position comes from embedding projections (UMAP, t-SNE, PCA)
- Dot color can represent:
  - mood (sentiment)
  - topic proxy
  - review group (cluster)
- Optional circle overlays summarize broader topic/cluster regions
- Selecting points enables local interpretation with examples and nearest neighbors

## Main Tabs

1. **Business Summary**
- High-level KPIs and distributions
- Topic, mood, trend, and wordcloud views for quick diagnosis

2. **Projection Map**
- Interactive 2D/3D semantic map
- Filter by date, topic, mood, cluster, and text query
- Review Deepdive panel for selected points
- Similar Reviews Explorer (kNN) for local neighborhood analysis

3. **Methods (Advanced)**
- Compact diagnostics for projection and clustering quality
- Intended for users who want methodological context

## Suggested User Workflow

1. Start in **Business Summary** to spot broad patterns.
2. Move to **Projection Map** and filter by period/topic/mood.
3. Select dots (click, lasso, or box) to inspect **Review Deepdive**.
4. Use **Similar Reviews Explorer** to see semantically close comments.
5. Export filtered rows for reporting.

## Data Scope

- Scope is fixed to **Olist** reviews.
- Filtered corpus size: **36,567 reviews**.
- Projection files are stacked across 3 models: **109,701 rows** total (`3 x 36,567`).
- English text shown in the UI comes from linked translated data files (not live translation calls).

## Key Features

- 2D and 3D projection map
- Color-by sentiment, topic, or cluster
- Optional region-circle overlays
- kNN similar-review retrieval
- Time-trend analytics and wordclouds
- CSV export of filtered table
- Optional PNG export of map snapshot

## Run Locally

From repository root:

```bash
pip install -r interactive-map-lite/requirements.txt
streamlit run interactive-map-lite/app.py
```

## Required Data Files

The app loads from repository-root `data/` and `artifacts/`:

- `data/projection_views/viz_{umap,tsne,pca}.csv` (preferred) or `.parquet`
- `data/projection_views/topic_proxy_lookup.csv` (preferred) or `.parquet`
- `data/projection_views/review_text_lookup.csv`
- `data/projection_views/wordcloud_terms.csv` (preferred) or `.parquet`
- `data/olist_order_reviews_dataset.csv`
- `data/olist_order_reviews_dataset_translated.csv`
- `artifacts/analysis-v1/<run>/semantic_map_points.csv`

## Interpretation Notes

- Region circles are **heuristic visual aids**, not strict model boundaries.
- Different projection methods can change map shape while preserving local structure differently.
- Topic proxy is rule-based and should be read as a weak thematic guide, not a full topic model.

## Troubleshooting

- If the app reports missing outputs, verify the required files above exist under `data/` and `artifacts/`.
- If snippet fields look empty, check `review_text_lookup.csv` and `review_id` consistency.
- If PNG export is unavailable, ensure `kaleido` is installed.
