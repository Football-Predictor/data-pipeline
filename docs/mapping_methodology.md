# Player‑matching methodology — step‑by‑step (accessible)

Purpose
- Explain, in practical terms, how StatsBomb player names are matched to FIFA player records and how that data is prepared for model training.
- Target reader: someone with a Computer‑Science background (basic ML/data engineering knowledge) but limited domain knowledge about football data sources.

Summary (one line)
- We use a layered, auditable pipeline that combines deterministic rules, token‑blocking, staged fuzzy matching, contextual validation (position checks), a small classifier, and human review to produce safe, reproducible player→FIFA mappings and training tables.

## Process diagram
```mermaid
flowchart TD
  A[StatsBomb player name] --> B[Normalize\n(lowercase, NFKD, strip punctuation)]
  B --> C{Exact normalized\nmatch?}
  C -- yes --> D[Accept mapping \n(write to player_map.csv)]
  C -- no --> E[Token‑blocking\n(build candidate set)]
  E --> F[Quick fuzzy\n(cheap filters)]
  F -- high --> G{Position check\n(GK / DEF / MID / FWD)}
  G -- pass --> D
  G -- fail --> H[Flag for review]
  F -- low --> I[Full fuzzy\n(token_sort/token_set)]
  I -- high --> J[Classifier (p ≥ th)]
  J -- yes --> D
  J -- no --> K[Targeted passes\n(bigrams / surname / initials)]
  K -- high --> D
  K -- low --> H
  H --> L[Human review CLI]\nL --> M[Manual accept / reject]
  D --> N[Audit log: method, score, timestamp]
  N --> O[Assign/ensure stable player_id (index+1 if missing)]
  O --> P[Build processed training tables\n(players_train / matches_train)]
  H --> Q[Mark as review/unmatched]
  P --> R[Model training / inference]
```

1) Inputs & outputs
- Inputs:
  - StatsBomb starting‑XI (matches_starting_players.parquet)
  - FIFA player table (fifa_players.parquet)
- Primary outputs (training‑ready):
  - `data/processed/players_train_1000.parquet` (one row per player, `player_id` + attributes)
  - `data/processed/matches_train_1000.parquet` (match rows with `home_starting_xi` / `away_starting_xi` referencing `player_id`)
- Secondary outputs: mapping files (`data/mappings/player_map.csv`, `player_map_review.csv`) and instrumentation (scores, methods).

2) Normalization (make strings comparable)
- Steps: lowercase → Unicode NFKD (strip diacritics) → remove punctuation/quotes → collapse whitespace. Example: `Jóse M. de‑la‑Cruz` → `jose m de la cruz`.
- Why: reduces surface variation so exact + fuzzy checks behave predictably.

3) Exact normalized lookup (fast, zero‑risk)
- If normalized StatsBomb name == normalized FIFA name → accept immediately.
- Captures straightforward cases and avoids unnecessary fuzzy work.

4) Token‑blocking (scale & precision)
- Split name into tokens (words). Use surname or other informative tokens to select a small candidate set from the large FIFA table.
- Purpose: reduce candidate size (from millions to a few dozen/thousand) so fuzzy scoring is fast and less likely to produce false positives.
- Safety: skip or down‑weight very common tokens (`de`, `da`, `jose`) using token caps/skip rules.

5) Staged fuzzy matching (increasingly permissive)
- Quick fuzzy: fast approximate scorers to filter obvious non‑matches.
- Full/exhaustive fuzzy: higher‑quality token‑aware scorers (token_sort_ratio, token_set_ratio) applied to blocked candidates.
- Targeted passes: bigrams (last two tokens), surname pass, initial+surname pass (handles `J. Martinez`), permissive fallback — each designed for specific failure modes.
- Each candidate is assigned a numeric score (0–100) and the top candidate(s) are recorded.

6) Contextual validation (rule‑based safety)
- Position‑group check: compare StatsBomb position (string) → group (GK/DEF/MID/FWD) to FIFA primary position group; require match for many auto‑accepts.
- Rationale: prevents obvious mismatches (e.g., matching a striker to a goalkeeper).

7) Classifier‑based promotion (probabilistic safety net)
- A small logistic classifier is trained on previously accepted mappings. Features:
  - `tok_sort`, `tok_set`, `partial`, `ratio` (string similarity scores)
  - `len_diff` (length difference)
  - `same_initials` (boolean)
  - `pos_match` (boolean)
- Use: predict whether a review candidate is correct; auto‑promote only at high probability.
- Current thresholds: `classifier` (high‑trust) ≈ p ≥ 0.90, `classifier_pos` (lower‑trust + often requires pos match) ≈ p ≥ 0.75.
- Result: classifier has safely promoted mappings (example: 191 promotions in the current run).

8) Human‑in‑the‑loop review
- All non‑high‑confidence candidates are written to `player_map_review.csv` with `candidate_fifa_id`, `score`, `method`, `status`.
- `scripts/review_mapping_cli.py` provides an interactive CLI for the human reviewer to accept/reject/replace candidates.
- Recommendation: prioritize reviewing players with high appearance counts or those that block match‑level completeness.

9) Auditing & reversibility
- Every mapping decision stores: method (which pass promoted it), score, timestamp/reviewer where applicable.
- Backups (`*.bak`) are kept when processed tables are overwritten.
- You can filter mappings by `method` (e.g., show `classifier_pos`) and revert as needed.

10) ID strategy & training tables
- If canonical `sofifa_id` is missing we assign a stable `player_id = index + 1` so models and the web app have a numeric reference.
- `build_training_tables.py` compiles player rows (attributes kept, NaNs preserved) and match rows where starting‑XI lists reference `player_id`.

11) Integrity checks & tuning
- `scripts/check_player_match_integrity.py` validates null/zero ids and starting‑XI completeness.
- `scripts/check_matches_full_match.py` reports how many matches have all players mapped (team‑level completeness). `scripts/simulate_threshold_coverage.py` helps decide promotion thresholds by simulating coverage vs risk.

12) Practical thresholds & recommended defaults
- Exact normalized match: accept.
- Fuzzy accept: prefer scores ≥ 85 + pos_match. Lower scores ⇒ review.
- Classifier: use p ≥ 0.90 as automatic accept; 0.75–0.90 ⇒ classifier_pos (consider review prioritization).
- Rationale: prioritize precision over recall because mapping errors introduce label noise downstream.

13) Monitoring & metrics you should track
- % player appearances mapped (per season / global). Current target: maximize while keeping precision high. Example: processed slice ≈ 93.2% appearances mapped.
- Matches fully mapped (both teams). Current processed slice ≈ 43.6% (improvable with country/team passes).
- Counts by `method` (how many were auto‑accepted by classifier, full_fuzzy, surname, etc.).

14) Known limitations & next steps
- Missing nationality in the FIFA CSV reduces confidence for international name collisions — add a nationality pass next.  
- Team‑context heuristics (consensus within a lineup) can further lower false positives.  
- Continue to keep a human review loop and sample mapping audits to estimate precision.

15) How to reproduce / quick commands
- Run mapping passes: `uv run scripts/match_players.py` / `scripts/match_players_fullfuzzy.py` / `scripts/match_players_position_pass.py`  
- Train classifier: `uv run scripts/train_mapping_classifier.py`  
- Apply classifier promotions: `uv run scripts/match_players_classifier_pass.py`  
- Review candidates interactively: `uv run scripts/review_mapping_cli.py`  
- Build processed training tables: `uv run scripts/build_training_tables.py`  
- Check integrity/coverage: `uv run scripts/check_player_match_integrity.py` and `uv run scripts/check_matches_full_match.py`.

Glossary (short)
- `token_set_ratio` / `token_sort_ratio` (tok_set / tok_sort): token‑aware fuzzy scores robust to order changes.  
- `partial_ratio`: substring match score (good for initials/short forms).  
- `ratio`: global character similarity.  
- `pos_match`: whether SB position group and FIFA primary group agree.

If you want, I can:
- add a short flowchart image or sequence diagram, or
- convert this into a one‑page cheat sheet you can include in the README.

---
Last updated: 2026-02-19
