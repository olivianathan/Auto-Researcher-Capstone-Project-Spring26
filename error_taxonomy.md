# Error Taxonomy — Auto-Researcher Agent Loop

Categorizes failures observed across iterations 1–5 logged in `experiment_log.csv`.

---

## 1. Signal Failure
*The loop ran but no meaningful improvement appeared.*

**Iter 2 — LR with class_weight='balanced' made both metrics worse.**
Adding balanced class weights was intended to improve minority-class recall and subgroup parity. Instead, AUC dropped (0.9242 → 0.9234), the equalized-odds gap increased to 1.0 (the worst possible value, up from 0.9767), and the objective fell from 0.7777 to 0.7734. The agent got a discard and extracted no useful signal from the intervention.

**Iters 1–5 — The fairness gap never came close to the target despite five iterations.**
The equalized-odds gap only moved from 0.9767 to 0.8202 across the entire loop — still 16× above the ≤ 0.05 target. The agent's strategy of switching model families (LR → RF → XGBoost) produced meaningful AUC gains but essentially no progress on the metric that was actually failing. The loop iterated without diagnosing the root cause.

---

## 2. Code Instability
*Crashes, inconsistent runs, or broken pipeline.*

**Iter 4 — Silent fallback could report metrics for the wrong model path.**
The notes for iter 4 explicitly document a fallback: "uses logistic fallback if XGBoost import fails." The try/except block in run.py would silently switch from scaled XGBoost to unscaled logistic regression if XGBoost threw an error, with no flag written to the log. If the fallback fired, the logged AUC of 0.9322 would reflect a completely different model than the one described.

**Two parallel log files existed simultaneously with incompatible schemas.**
`results.tsv` (written by the restructured run.py) and `experiment_log.csv` (written by the notebook loop) existed at the same time with different column sets. The `discard_or_keep` logic in run.py read from `results.tsv`, while the actual history lived in `experiment_log.csv`. Any iteration that ran through run.py would have compared against an empty log and always returned `keep`, breaking the sequential gating entirely.

---

## 3. Evaluation Leakage
*Metric improved but comparability was compromised.*

**The equalized-odds gap was computed differently in the notebook vs. run.py.**
The notebook's `equalized_odds_gap()` helper computed per-axis gaps (sex, race, age separately) and was stricter in how it handled small subgroups. The original run.py used a flat loop over all groups combined and only tracked TPR gap. Iter 1–5 values in the log were produced by the notebook implementation; any future run through run.py (before today's fix) would produce a different number for the same model, making cross-iteration comparison invalid.

**The old agent loop re-split the data from scratch instead of using the locked splits.**
The checkpoint (`03_agent_loop.py`) loaded `nhanes_clean.csv` and called `train_test_split` directly rather than reading the locked `X_train.csv` / `X_val.csv` written by `prepare.py`. Even with `random_state=42`, differences between `nhanes_clean.csv` and `nhanes_selected.csv` (the source for the locked splits) could produce a different validation set. Metrics across iterations are therefore not guaranteed to be evaluated on the same rows.

---

## 4. Agent Misbehavior
*Agent ignored rules or made out-of-scope changes.*

**The agent ran its entire experiment loop inside the notebook instead of modifying model.py.**
The contract was: agent modifies `model.py`, calls `run.py` to evaluate. Instead, the agent wrote all five experiments directly into Capstone.ipynb cell 7 as a hardcoded `experiments` list with inline model definitions, data loading, and evaluation logic. It never touched `model.py` and never called `run.py`. The frozen-file boundary was bypassed entirely.

**The agent wrote directly to experiment_log.csv from within the notebook loop, bypassing run.py.**
`run.py` was designed as the single point of truth for logging. The agent instead appended rows to `experiment_log.csv` from inside the notebook's evaluation loop, meaning the log contains entries that correspond to no particular state of `model.py`. If `model.py` had been modified between notebook runs, the log would have no record of which version of the model produced which result.
