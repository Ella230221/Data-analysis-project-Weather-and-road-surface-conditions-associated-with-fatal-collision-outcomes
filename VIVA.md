# Viva and live demonstration

## Suggested five-minute sequence

1. **0:00–0:45 · Data and question.** Explain the difference between a source row and a unique collision. Open `results/audit.json` and point to the fatal prevalence.
2. **0:45–1:45 · Spark processing.** Open `analysis.py`. Show the composite-key `groupBy`, distinct-value checks, eligibility filter and weather–surface aggregation. Show the actual grouped output in the notebook. During a live run, the terminal prints the Spark summary before the longer timed block.
3. **1:45–2:35 · Environmental results.** Show the fatal-share plot. Read a rate and its denominator. Explain why the denominator is recorded injury or fatal collisions and how small groups affect interval width.
4. **2:35–3:45 · Model evaluation.** Show the holdout table and precision–recall plot. Explain M0, M1 and M2, the common test set, and why average precision is the primary comparison.
5. **3:45–5:00 · Timing and conclusion.** Show all measured runs and median/range. Describe which Spark settings changed, what the timer includes and how equality was checked. Close with the answer to each research question.

Have the verified results open in the notebook before presenting. If time permits, rerun `analysis.py` against the local CSV and show the printed summary and regenerated outputs.

## Team contributions and live handoff

| Team member | Project contribution | Evidence and suggested demo role |
| --- | --- | --- |
| Xiaoshuang Yang | Data audit, code mappings, descriptive analysis and model pipeline | Audit log, environmental figure, model metrics and sensitivity output. Explain collision definition, descriptive results and model evaluation. |
| Ruxue Yang | PySpark preparation, performance experiment and timing analysis | Spark aggregation and benchmark functions, grouped output, timing log and figure. Explain Spark validation, partitions and benchmark controls. |

Both team members share result interpretation, validation, report integration and presentation preparation. Each should be able to explain every step.

## Questions to rehearse

- **Why does a case ID repeat?** The CSV has person and vehicle records. A composite year-and-case key identifies collisions.
- **Why accuracy and AP tell different stories?** Fatal collisions are rare. Predicting the majority class does well on accuracy while recovering no fatal cases. AP evaluates the ranking of fatal cases across thresholds.
- **Why scikit-learn?** The approved proposal specifies L2 logistic regression on a small collision-level table, with an interpretable nested feature comparison. Spark handles the raw processing.
- **What can the weather percentages establish?** They describe conditional severity among collisions in the release. Travel exposure and causal factors are outside the data.
- **What exactly did Spark time?** The same uncached read-to-aggregate computation including explicit repartitioning and collection. JVM startup and the warm-up runs are outside the recorded times.
