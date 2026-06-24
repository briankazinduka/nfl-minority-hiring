# NFL Coaching Career Movement & Hiring Equity

Predicting an NFL coach's **next-season career movement** — *Promotion, Sideways, Demotion, or Exit/Gap* — from role, team performance, experience, and demographics, and auditing whether the model's accuracy differs by race.

This repository started as a Northeastern University introductory machine-learning **final project** and was subsequently revised through an in-depth code review. It now contains three generations of the analysis plus the original write-up.

---

## Project context

The original class project asked a socially meaningful question: *can we predict how NFL coaches move through the profession, and does a model trained on historical data treat Black and White coaches' trajectories with equal accuracy?* This connects to long-running discussion about minority representation in NFL coaching (e.g., the Rooney Rule) — hence the repository name.

The pipeline:
1. Joins a **coaching roster** (who coached what role for which team, by year) to **team performance stats**.
2. Builds a **tier hierarchy** of coaching roles (position coach → coordinator → head coach) and labels each coach-season by where they move next.
3. Trains a classifier to predict that movement, then **audits prediction accuracy by race**.

---

## Repository contents

| File | Purpose |
|---|---|
| **`ML_project_final.ipynb`** | **v1 — the original class submission.** The analysis as turned in. Kept for provenance; it contains the methodological issues the later versions fix. |
| **`ML_project_final_v2.ipynb`** | **v2 — leakage-corrected revision.** Same overall analysis and multi-model breadth (RF / Decision Tree / Naive Bayes / KNN / LDA, class-weight + SMOTENC imbalance handling, MDI + permutation importance), but with the data-leakage and labeling bugs fixed. |
| **`ML_project_final_v3.ipynb`** | **v3 — focused rebuild and the recommended version.** A tight, fully annotated argument: it demonstrates how *naive* preprocessing produces a high-accuracy-but-useless model, adds live 2024–25 data, engineers and selects features, and runs a rigorous fairness audit with confidence intervals. **Start here.** |
| **`Machine_Learning_Final_Project_Paper.pdf`** | The written report accompanying the original (v1) project. |
| **`coach list 89-25 with race.dta`** | Coaching roster (Stata format): coach name, DOB, team, year, role, race — 1989–2026. |
| **`team_stats_2003_2023.csv`** | Season-level team performance (Pro-Football-Reference schema), 2003–2023. v3 extends this to 2024–2025 live via `nflreadpy`. |

---

## What the review changed, and why

The revisions targeted issues that made the original results look better than they really were, then rebuilt the analysis around an honest demonstration of that gap.

| # | Issue found | Fix (in v2 / v3) |
|---|---|---|
| 1 | **Train/test leakage** — a random split scattered the *same coach* across train and test, inflating every score. | Group-aware split on `CoachID` (no coach in both sets). |
| 2 | **Feature leakage** — `statistical_positionality` and `prev_league_churn` were computed over the full dataset, leaking test info. | Recompute these pooled features from **training rows only**. |
| 3 | **Right-censoring** — the most recent season was mislabeled `Exit/Gap` for everyone (no "next year" to observe). | Compute labels with the final season present, then drop that season from modeling. |
| 4 | **Accuracy mirage** — ~73% accuracy looked strong but came from always guessing the majority class (`Sideways` ≈ 70%). | Report **balanced accuracy** + per-class recall; show that adding `class_weight` *alone* does **not** fix it — only the full principled pipeline does. |
| 5 | **Arbitrary tier definition** — bottom tiers didn't reflect real promotion likelihood. | Tiers derived from **cross-era promotion rates** (1989–2002 and 2003–2025): QB above other position coaches; LB/DB the next most promotion-prone. |
| 6 | **Stale data** — team stats stopped at 2023, so 2024–25 coaches got median-imputed (fake) numbers. | Pull 2024–25 game results live from `nflreadpy`; leave only the un-reproducible PFR-only columns to imputation. |
| 7 | **No feature engineering / selection narrative.** | Engineered `role_tenure`, `prev_movement`, `win_trend`, `age`; selected the final set via F-scores, ablation, and permutation importance. |
| 8 | **Fairness claims had no uncertainty.** | Report group sizes and **95% bootstrap confidence intervals** on per-race accuracy and the gap. |

### Headline v3 results
- **Naive** model: accuracy **0.73**, balanced accuracy **0.36** — basically predicts "Sideways" for everyone (promotion recall ≈ 1%).
- **Naive + imbalance handling only**: essentially unchanged → the problem is **not** just class imbalance.
- **Principled** (full preprocessing): accuracy **0.56** (honest), balanced accuracy **0.45** — actually identifies promotions/demotions.
- **Feature selection**: balanced accuracy **0.446 → 0.491** with the final 10-feature set (dropping `score_pct`, `prev_league_churn`, `age`; adding `role_tenure`, `prev_movement`, `win_trend`).
- **Fairness**: a per-race accuracy gap exists and is statistically reliable (CI excludes 0), but it is **not** driven by the model using race directly — removing `Race` as a feature changes balanced accuracy by only +0.001.

---

## Running it

Requires Python 3.11+ and:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn nflreadpy nbformat
```

Then, from this folder:

```bash
jupyter notebook ML_project_final_v3.ipynb   # Kernel → Restart & Run All
```

> **Note:** v3 downloads 2024–25 NFL data via `nflreadpy`, so it needs an internet connection on first run.

---

## Subsequent steps / future work

- **Minority-class precision.** The principled model now *finds* promotions (high recall) but with many false alarms (macro-F1 stays modest at ~0.38). Improving precision is the main open modeling problem.
- **Metric stability.** Results come from a single group-aware split; repeated splits or nested cross-validation would put error bars on the headline numbers.
- **Carry over v2's breadth into v3** (optional): the multi-algorithm comparison, the SMOTENC oversampling experiment, and the "optimistic vs. pessimistic" error-direction bias breakdown live in v2 but were left out of v3 to keep its argument tight.
- **2026 season** can be ingested once it is played and `nflreadpy` has results.
- **`prev_league_churn` lag** — the feature was dropped for lack of signal, but if revived, confirm the intended one-vs-two-season lag.

---

## Attribution

The original project, analysis, and paper are the author's (a Northeastern ML course final project). The v2/v3 revisions, the data-leakage and labeling fixes, the feature engineering/selection, and this README were produced collaboratively with **Claude Code** as a code-review and improvement exercise.
