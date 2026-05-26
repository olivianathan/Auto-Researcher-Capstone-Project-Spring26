# Error Taxonomy -- Auto-Researcher Agent Loop

Categorizes failures observed across iterations 1-14 logged in `experiment_log.csv`.

---

## 1. Signal Failure
*The loop ran but no meaningful improvement appeared.*

**Iter 2 -- LR with class_weight='balanced' made both metrics worse.**
Adding balanced class weights was intended to improve minority-class recall and subgroup parity. Instead, AUC dropped (0.9242 -> 0.9234), the equalized-odds gap increased to 1.0 (the worst possible value, up from 0.9767), and the objective fell from 0.7777 to 0.7734. The agent got a discard and extracted no useful signal from the intervention.

**Iters 1-6 -- Six iterations of model-family switching never diagnosed why age fairness was stuck.**
The equalized-odds gap for age only moved from 0.9767 to 0.7876 across six iterations (LR -> RF -> XGBoost -> regularized XGBoost). The agent kept changing model architecture when the real problem was the metric: equalized TPR across age groups is structurally impossible for a disease risk model because high-risk conditions are genuinely rare in children and common in elderly patients. Switching to MACE at iter 7 resolved in one step what six iterations could not -- the fairness metric dropped from 0.79 to 0.04 immediately.

**Iter 9 -- Random Forest + calibration was discarded because race EqOdds regressed.**
RF calibrated with isotonic regression achieved better age MACE (0.0188) than iters 7-8, but the race equalized-odds gap jumped to 0.073, exceeding the 0.05 target and dragging the overall metric and objective below the kept threshold. The calibration layer helped age but hurt race parity.

**Iters 10-11 -- Confounded experiments changed multiple variables simultaneously.**
Iter 10 changed both the XGBoost architecture (depth=5, n=500, lr=0.03) and the base model relative to the calibration baseline (iter 7). Iter 11 then changed XGBoost params again AND cv folds (3->5) in the same run. Both were logged as improvements (obj 0.9224, 0.9228), but it was impossible to attribute the gain to any single change. The underlying question -- does cv=5 beat cv=3 on the same base model? -- remained unanswered until the controlled re-runs in iters 12-14.

**Iter 14 -- Two-stage age-quintile calibration hurt race fairness.**
Adding a per-age-quintile isotonic second pass on top of the global isotonic calibrator was expected to improve age MACE further. Instead, AUC dropped from 0.9285 to 0.9191 and the race equalized-odds gap jumped from 0.0229 to 0.0619, pushing overall fairness above the 0.05 target. The quintile-specific calibrators over-fit the age-group boundaries, distorting probability estimates in ways that affected race subgroups disproportionately.

---

## 2. Code Instability
*Crashes, inconsistent runs, or broken pipeline.*

**Iter 4 -- Silent fallback could report metrics for the wrong model path.**
The notes for iter 4 explicitly document a fallback: "uses logistic fallback if XGBoost import fails." The try/except block in run.py would silently switch from scaled XGBoost to unscaled logistic regression if XGBoost threw an error, with no flag written to the log. If the fallback fired, the logged AUC of 0.9322 would reflect a completely different model than the one described.

**Two parallel log files existed simultaneously with incompatible schemas.**
`results.tsv` (written by the restructured run.py) and `experiment_log.csv` (written by the notebook loop) existed at the same time with different column sets. The `discard_or_keep` logic in run.py read from `results.tsv`, while the actual history lived in `experiment_log.csv`. Any iteration that ran through run.py would have compared against an empty log and always returned `keep`, breaking the sequential gating entirely.

**Iters 7-11 -- Unicode encoding crash took down the pipeline on first real run.**
Both `prepare.py` and `run.py` contained box-drawing and arrow characters (─, ->, <=, --) in print statements that were valid in Jupyter's UTF-8 environment but caused `UnicodeEncodeError` on Windows (cp1252 codec). The pipeline crashed before logging a single result. A botched encoding-replacement pass then corrupted section headers in `prepare.py` to `??` strings, requiring a full rewrite of the file before any experiment could run.

**Schema mismatch on first run.py execution added a 20th field to a 19-column CSV.**
Adding `improved_over_baseline` to run.py's `log_columns` without updating the existing `experiment_log.csv` header caused pandas to raise "Expected 19 fields in line 7, saw 20" when the performance plot tried to read the log. The plot update silently failed and the bad row had to be stripped manually before the next run.

---

## 3. Evaluation Leakage
*Metric improved but comparability was compromised.*

**The equalized-odds gap was computed differently in the notebook vs. run.py.**
The notebook's `equalized_odds_gap()` helper computed per-axis gaps (sex, race, age separately) and was stricter in how it handled small subgroups. The original run.py used a flat loop over all groups combined and only tracked TPR gap. Iter 1-5 values in the log were produced by the notebook implementation; any run through run.py before the fix would produce a different number for the same model, making cross-iteration comparison invalid.

**The old agent loop re-split the data from scratch instead of using the locked splits.**
The checkpoint (`03_agent_loop.py`) loaded `nhanes_clean.csv` and called `train_test_split` directly rather than reading the locked `X_train.csv` / `X_val.csv` written by `prepare.py`. Even with `random_state=42`, differences between `nhanes_clean.csv` and `nhanes_selected.csv` (the source for the locked splits) could produce a different validation set. Metrics across iterations are not guaranteed to be evaluated on the same rows.

**Iters 7-14 -- The `eq_odds_gap_age` column stores two different quantities with no flag.**
For iters 1-6 the column holds equalized-odds TPR gap for age groups. For iters 7-14 it holds MACE (mean absolute calibration error). Both are in [0,1] but measure fundamentally different things -- a drop from 0.82 to 0.03 across the iter 6/7 boundary looks like progress in the log but is partly a metric definition change, not a model improvement. There is no column in the log explicitly flagging which definition applies to each row, making any automated trend analysis across the full 14 iterations misleading.

---

## 4. Agent Misbehavior
*Agent ignored rules or made out-of-scope changes.*

**The agent ran its entire experiment loop inside the notebook instead of modifying model.py.**
The contract was: agent modifies `model.py`, calls `run.py` to evaluate. Instead, the agent wrote all five experiments directly into Capstone.ipynb cell 7 as a hardcoded `experiments` list with inline model definitions, data loading, and evaluation logic. It never touched `model.py` and never called `run.py`. The frozen-file boundary was bypassed entirely.

**The agent wrote directly to experiment_log.csv from within the notebook loop, bypassing run.py.**
`run.py` was designed as the single point of truth for logging. The agent instead appended rows to `experiment_log.csv` from inside the notebook's evaluation loop, meaning the log contains entries that correspond to no particular state of `model.py`. If `model.py` had been modified between notebook runs, the log would have no record of which version of the model produced which result.

**Iters 7-11 -- Unicode characters were written into frozen .py files despite a Windows runtime.**
When restructuring the pipeline, `run.py` and `prepare.py` were written with UTF-8 box-drawing and arrow characters (─, ->, <=) that rendered correctly in the Jupyter notebook environment but are outside the Windows cp1252 character set. Writing environment-incompatible characters into frozen pipeline files is an out-of-scope change -- it modified files the agent is not supposed to touch and broke the pipeline for any non-notebook execution context.
