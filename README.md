# Reproducing the Pipeline

This document explains how to reproduce the full data pipeline for this project, from downloading the raw Amazon Reviews 2023 data to producing the train/test splits used by every recommendation model.

The pipeline has four steps:

1. Download the raw data.
2. Run the `load_*` scripts to load and inspect reviews.
3. Run the `extract_*` scripts to extract and verify item metadata.
4. Run the cleaning and EDA notebook to produce the train/test splits.

All commands below should be run from the repository root:
