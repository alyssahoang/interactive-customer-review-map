# Olist Negative Sentiment Geometry

This repository contains a reproducible NLP project that studies how negative customer reviews are arranged in embedding space for Brazilian Portuguese e-commerce data (Olist).

## Project Objective

The project compares three embedding families under one shared pipeline:
- BERTimbau (Portuguese monolingual)
- Multilingual MiniLM
- OpenCLIP text encoder

Main questions:
- Does negative sentiment form dense geometric pockets or stay diffuse across topics?
- How does this pattern change by embedding model?

## Repository Map

- `notebook/`
- `notebook/olist_negative_sentiment_geometry_analysis.ipynb`: main analysis notebook (demo/orchestration).
- `report/`
- `report/olist_negative_sentiment_geometry_report.tex`: report source.
- `report/olist_negative_sentiment_geometry_report.pdf`: compiled report.
- `report/olist_negative_sentiment_geometry_assets/`: figures used by the report.
- `src/`: reusable Python modules (data, preprocessing, embeddings, projection, clustering, evaluation, helpers).
- `scripts/`: reproducibility scripts and pipeline entrypoints.
- `interactive-map-lite/`: Streamlit app for stakeholder exploration.
- `data/`: source data + processed and projection files.
- `artifacts/`: cached analysis outputs.

## Data Scope

- Filtered Olist review corpus: `36,567` reviews.
- Projection views are stacked across 3 models: `109,701` points (`3 x 36,567`).
- English snippets come from linked translated dataset files, not live translation.

## Reproducibility Setup

From repository root:

```bash
pip install -r requirements.txt
```

## Reproduce Analysis Outputs

1. Validate package completeness:
```bash
python scripts/run_submission_pipeline.py --validate-only
```

2. Rebuild projection views (fast path from shipped caches):
```bash
python scripts/run_submission_pipeline.py --skip-report-pack
```

3. Rebuild report-pack tables/figures:
```bash
python scripts/run_submission_pipeline.py --skip-projection
```

4. Run full pipeline:
```bash
python scripts/run_submission_pipeline.py
```

## Run Notebook

```bash
jupyter notebook notebook/olist_negative_sentiment_geometry_analysis.ipynb
```

Recommended for quick reruns inside notebook:
- `PHASE3_RECOMPUTE = False`
- `RUN_HEAVY_3D = False`

## Compile Report

```bash
cd report
latexmk -pdf -interaction=nonstopmode -file-line-error olist_negative_sentiment_geometry_report.tex
```

## Run Interactive App

```bash
pip install -r interactive-map-lite/requirements.txt
streamlit run interactive-map-lite/app.py
```

For app-specific details, see `interactive-map-lite/README.md`.
