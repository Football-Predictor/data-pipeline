# Mapping Findings & Actions ✅

**TL;DR:** Multi-pass mapping + classifier and targeted heuristics substantially improved coverage and safety. Current workspace state: **3,661 accepted mappings**, `player_map_review.csv` (1k subset) **1,942 review rows**, **50,114 / 53,761 player appearances mapped (93.2%)**, and **1,066 matches fully mapped (43.6%)**. We also produced a 1,000-match processed training slice and assigned stable index-based `fifa_id` values so every player has an ID for inference.

---

## What I implemented (scripts & passes) 🔧
- `scripts/match_players.py` — starting-XI extraction, normalization, initial exact + quick-fuzzy pass
- `scripts/match_players_fullfuzzy.py` — exhaustive fuzzy pass (token blocking, multi-scorer)
- `scripts/match_players_position_pass.py` — position-aware promotion pass
- `scripts/match_players_bigrams_fuzzy.py` — bigram token intersection pass (targeted)
- `scripts/match_players_surname_fuzzy.py` / `scripts/match_players_initial_surname.py` — surname / initial+surname passes
- `scripts/match_players_permissive_fuzzy.py` — permissive/fallback fuzzy attempts
- `scripts/train_mapping_classifier.py` + `scripts/match_players_classifier_pass.py` — small classifier to safely promote review candidates
- `scripts/review_mapping_cli.py` — interactive reviewer for manual validation
- `scripts/restrict_to_first_n_matches.py` — create a 1k-match subset (mapping + FIFA subset)
- `scripts/build_training_tables.py` — build `players_train_1000.parquet` / `matches_train_1000.parquet`
- `scripts/assign_sofifa_index_ids.py` / `scripts/force_index_ids.py` — assign stable index-based IDs and reconcile processed tables
- Diagnostics & helpers: `scripts/check_mapping_stats.py`, `scripts/check_matches_full_match.py`, `scripts/check_player_match_integrity.py`, `scripts/analyze_unmatched.py`, `scripts/inspect_fifa_1000.py`

All scripts are runnable via `uv run scripts/<script>.py` (they include inline dependency metadata).

---

## Key metrics (current) 📊
- Unique StatsBomb players (global): **4,472**
- Auto-accepted mappings (global): **3,661** (`data/mappings/player_map.csv`)
- Review / ambiguous rows (current review file = 1k subset): **1,942** (`data/mappings/player_map_review.csv`)
- Player appearances matched (using accepted map): **50,114 / 53,761 (93.2%)**
- Matches where BOTH teams are fully matched: **1,066 / 2,445 (43.6%)**

Processed training slice (first 1,000 matches):
- `data/processed/players_train_1000.parquet` — **1,942** players
- `data/processed/matches_train_1000.parquet` — **1,000** matches

Simulation notes:
- Classifier + position-aware promotions produced the largest safe gains; targeted surname/bigrams passes recovered additional edge cases.
- Lowering thresholds (e.g., auto-accept ≥75) yields much higher match-level coverage but increases risk and manual review burden.

---

## What we tried & observed 🧭
- Exact normalized matching alone provided very low coverage (many names have diacritics/variants).
- Quick fuzzy (token narrowing) improved matches but was limited by candidate-sizes and token commonness.
- Full fuzzy (exhaustive but blocked) found many safe matches — added ~2,445 accepted mappings.
- Position-aware promotions added **905** more safe mappings by validating FIFA position groups.

Performance & scale notes:
- FIFA parquet has ~10M rows; building indices and tokenization is CPU & memory heavy. Token-blocking + caps kept runs practical on CPU.
- Some pandas dtype issues (nullable ints vs float columns) required careful dtype handling when writing review updates.

---

## Issues & edge cases encountered ⚠️
- Name variants: middle names, suffixes, diacritics, apostrophes, and reordered names cause lots of ambiguity.
- Token common tokens like "de", "da" or first names require skipping or de-prioritizing.
- Lack of nationality/country alignment reduces confidence for international names with common tokens.
- A few scripts initially referenced variables in the wrong scope and needed fixes (now committed).

---

## Recommendations / Next steps (prioritized) 📈
1. **Add nationality/country-aware promotions** — high-impact (match SB `player_country` to FIFA `nationality` / `country` before promoting). 🌍
2. **Team / context heuristics** — promote ambiguous players when teammates/positions show consistent mapping in the same match.
3. **Train a small classifier** to predict correct mapping (features: fuzzy score(s), position match, country match, token overlap, team-context counts). Use it to safely auto-accept with a precision target. 🧠
4. **Add `scripts/review_mapping.py`** — single-reviewer CLI (page-by-page) to accept/reject rows in `player_map_review.csv`.
5. **Document coverage per season and set a cutoff season** for training so model sees a consistent mapping snapshot.

Optional: If you want immediate >80% match-level coverage, use fallback fills per position (e.g., use position means) — but this will introduce synthetic approximations.

---

## Label sources & label‑noise risk 🏷️
- Labels (match outcomes, goals) are taken from StatsBomb `matches.parquet` and are considered authoritative for final scores; however **label noise can be introduced by mapping errors** (incorrect player→FIFA matches) and by timing mismatches where FIFA attribute snapshots do not align with the match date (temporal drift). Substitutions, lineup errors, or missing starting XI rows in StatsBomb can also create noisy inputs that look like label noise at model training time.
- Mitigations we use: keep mapping status (`accepted` / `review` / `unmatched`) and only promote high‑confidence matches (fuzzy+position checks, classifier thresholds); flag and hold `review` rows for manual inspection via `review_mapping_cli.py`; run sensitivity analyses to measure model robustness to mapping errors before production deployment.

## Key data assumptions 🔎
- FIFA attributes (pace, shooting, passing, dribbling, defending, physic) are treated as proxies for a player’s ability at or near the match date — we assume temporal drift is small within our chosen training window or is handled via season cutoff.
- Starting XI sufficiently represents team starting strength for match‑level prediction (we assume missing/unmapped players are rare or handled by documented fallbacks). 
- Mapping corrections (classifier promotions, position checks, manual review) keep mapping error rates low enough that label noise remains manageable (current processed slice: ~93.2% appearance coverage).
- Where canonical `sofifa_id` is absent we use a stable index‑based `player_id` for reproducible joins; canonical IDs should replace these when available.

---

## Repro & commands 🧪
- Check stats: `uv run scripts/check_mapping_stats.py` 
- Run exhaustive fuzzy pass: `uv run scripts/match_players_fullfuzzy.py`
- Run position-aware promotions: `uv run scripts/match_players_position_pass.py`
- Re-simulate thresholds: `uv run scripts/simulate_threshold_coverage.py`
- Compute fully-mapped matches: `uv run scripts/check_matches_full_match.py`

---

## Changelog (short)
- **2026-02-19** — Converted processed tables to a single index-based `player_id`; updated `matches_train_1000.parquet` to reference `player_id`; added `convert_processed_to_index_only.py`; verified integrity. ✅
- **2026-02-16** — Added classifier promotion pass, bigram/surname/initial passes, review CLI; created 1k-match processed slice and assigned stable index-based IDs to make the dataset inference-ready. Coverage: **93.2%** appearance, **43.6%** fully-mapped matches.
- **2026-02-01** — Implemented multi-pass mapping pipeline (exact → quick-fuzzy → full-fuzzy → position-aware); ran exhaustive fuzzy and position promotions; added simulation tooling and diagnostics.

If you'd like, I can:
- Add a `CHANGELOG.md` file with these entries and commit history, or
- Open a pull request with these docs + scripts changes and a short summary for reviewers.

---

Last updated: 2026-02-19
