# Operational & pipeline scripts

One-off operational, data-generation, and training scripts. They are **not**
imported by the API service (`main.py`) — run them manually.

> **Run from the repo root** (e.g. `python scripts/seed_test_session.py`). Each
> script resolves the repo root for imports, but the data paths some of them use
> (`data/…`) are relative to your current directory.

## Local dev / testing
| Script | Purpose |
|---|---|
| `seed_test_session.py` | Create a throwaway user + a real `session` token for Postman/curl testing. `python scripts/seed_test_session.py <userId> [--cf <h>] [--lc <h>] [--days N]` |

## Qdrant (vector store) ops
| Script | Purpose |
|---|---|
| `check_qdrant.py` | Inspect what's in the Qdrant collection and why cold-start returns empty. |
| `check_titles.py` | Sanity-check problem titles/slugs in the store. |
| `create_qdrant_indexes.py` | Create the payload indexes the recommender queries. |
| `reset_qdrant.py` | Drop/recreate collections (destructive). |

## Offline data & ranking pipeline
| Script | Purpose |
|---|---|
| `run_full_pipeline.py` | End-to-end offline pipeline (ingest → embed → graph → Qdrant/Neo4j). `--input-json`, `--force-offline`, `--skip-offline`, `--no-neo4j`. |
| `generate_dataset.py` | Build the LightGBM training dataset (`lightgbm_dataset.jsonl`). |
| `export_difficulty.py` | Export the per-problem difficulty map (`real_difficulty_map.json`). |
| `regenerate_graph_artifacts.py` | Rebuild the problem/topic graph JSON artifacts from source. |
| `train_ranker.py` | Train the LightGBM ranking model from the generated dataset. |

Generated outputs (`lightgbm_dataset.jsonl`, `real_difficulty_map.json`, the
25 MB `1000_manifest_final_slugs_filled.json`) are git-ignored — they are
regenerated, not committed.
