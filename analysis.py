"""Reproduce the 2019 collision analysis and local Spark benchmark."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.ticker import PercentFormatter
from pyspark.sql import SparkSession, functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

KEY = ["C_YEAR", "C_CASE"]
FIELDS = ["C_SEV", "C_WTHR", "C_RSUR", "C_HOUR", "C_MNTH", "C_RCFG", "C_VEHS"]
ENV = ["C_WTHR", "C_RSUR"]
CONTROL = ["hour_band", "C_MNTH", "C_RCFG", "vehicle_group"]
SPECIAL = {"Q", "QQ", "U", "UU", "X", "XX", "Missing", "Invalid"}
LABELS = {
    "C_WTHR": dict(zip("1234567QUX", ["Clear", "Overcast", "Rain", "Snowfall", "Freezing rain / sleet / hail",
                                      "Limited visibility", "Strong wind", "Other", "Unknown", "Not reported"])),
    "C_RSUR": dict(zip("123456789QUX", ["Dry", "Wet", "Fresh snow", "Slush", "Ice / packed snow",
                                         "Debris", "Muddy", "Oil", "Flooded", "Other", "Unknown", "Not reported"])),
}
RANGES = {"C_WTHR": (1, 7), "C_RSUR": (1, 9), "C_HOUR": (0, 23), "C_MNTH": (1, 12),
          "C_RCFG": (1, 12), "C_VEHS": (1, 99)}
REFERENCES = dict(zip(CONTROL + ENV, ["12–15", "1", "1", "2", "1", "1"]))


def read_source(spark, path):
    return spark.read.option("header", True).option("mode", "FAILFAST").csv(str(path))


def collision_groups(raw):
    selected = raw.select(*[
        F.when(F.trim(F.col(c)).rlike(r"^[0-9]+$"),
               F.regexp_replace(F.trim(F.col(c)), r"^0+(?=[0-9])", ""))
        .otherwise(F.coalesce(F.trim(F.col(c)), F.lit("Missing"))).alias(c)
        for c in KEY + FIELDS
    ]).filter((F.col("C_YEAR") == "2019") & F.col("C_CASE").rlike(r"^[0-9]+$"))
    return selected.groupBy(*KEY).agg(
        *[F.min(c).alias(c) for c in FIELDS],
        *[F.countDistinct(c).alias(c + "_versions") for c in FIELDS],
    )


def eligible(groups):
    from functools import reduce
    from operator import and_
    consistent = reduce(and_, (F.col(c + "_versions") == 1 for c in FIELDS))
    table = groups.filter(consistent & F.col("C_SEV").isin("1", "2")).select(*KEY, *FIELDS)
    for c, (low, high) in RANGES.items():
        value = F.col(c)
        valid = value.cast("int").between(low, high) | value.isin(*sorted(SPECIAL))
        table = table.withColumn(c, F.when(valid, value).otherwise("Invalid"))
    return table.withColumn("fatal", (F.col("C_SEV") == "1").cast("int"))


def joint_summary(table):
    return (table.groupBy(*ENV).agg(F.count("*").alias("n"), F.sum("fatal").alias("fatal"))
            .orderBy(*ENV))


def descriptive(joint, out):
    records = []
    for field in ENV:
        counts = joint.groupby(field)[["n", "fatal"]].sum()
        for code, (n, f) in counts.iterrows():
            p, z = f / n, 1.95996398454
            d = 1 + z*z/n
            midpoint = (p + z*z/(2*n))/d
            half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
            records.append([field, code, LABELS[field].get(code, code), n, f, p,
                            max(0, midpoint-half), min(1, midpoint+half), n < 100])
    summary = pd.DataFrame(records, columns=["field", "code", "category", "n", "fatal", "rate",
                                             "lower95", "upper95", "sparse"])
    summary.to_csv(out / "environment.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharex=True, layout="constrained")
    for ax, (field, title) in zip(axes, zip(ENV, ["Weather", "Road surface"])):
        plot = summary[(summary.field == field) & ~summary["sparse"] & ~summary.code.isin(SPECIAL)]
        ax.errorbar(plot.rate, range(len(plot)), xerr=[plot.rate-plot.lower95, plot.upper95-plot.rate],
                    fmt="o", color="#006c67", capsize=4)
        ax.set_yticks(range(len(plot)), [f"{row.category}  ({row.n:,})" for row in plot.itertuples()], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(joint.fatal.sum()/joint.n.sum(), color="#6e7681", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Fatal share and 95% Wilson interval")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=.2)
    fig.savefig(out / "environment.png", dpi=180)
    plt.close(fig)
    return summary


def fit_models(train, test, out, suffix="", class_weight=None):
    y_train, y_test = train.fatal, test.fatal
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    results, scores, coefficients = [], {}, []
    for name, features in [("M0", []), ("M1", CONTROL), ("M2", CONTROL + ENV)]:
        threshold, selected_c = 0.5, None
        if features:
            pipe = make_pipeline(
                OneHotEncoder(drop=[REFERENCES[c] for c in features], handle_unknown="ignore"),
                LogisticRegression(solver="lbfgs", max_iter=1000, class_weight=class_weight),
            )
            search = GridSearchCV(pipe, {"logisticregression__C": [.1, 1, 10]},
                                  scoring="average_precision", cv=folds, error_score="raise")
            search.fit(train[features], y_train)
            model = search.best_estimator_
            selected_c = search.best_params_["logisticregression__C"]
            oof = cross_val_predict(model, train[features], y_train, cv=folds,
                                    method="predict_proba")[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_train, oof)
            f1 = 2 * precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
            threshold = float(thresholds[np.argmax(f1)])
            prob = model.predict_proba(test[features])[:, 1]
            fitted = model.named_steps["logisticregression"]
            coeff = fitted.coef_[0]
            assert max(fitted.n_iter_) < 1000 and np.isfinite(coeff).all()
            names = model.named_steps["onehotencoder"].get_feature_names_out(features)
            coefficients.extend((name, f, c, np.exp(c)) for f, c in zip(names, coeff))
        else:
            prob = np.full(len(test), y_train.mean())
        prediction = prob >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        results.append(dict(model=name, train_n=len(train), test_n=len(test), test_fatal=int(y_test.sum()),
                            C=selected_c, threshold=threshold, AP=average_precision_score(y_test, prob),
                            AUC=roc_auc_score(y_test, prob), Brier=brier_score_loss(y_test, prob),
                            accuracy=accuracy_score(y_test, prediction), precision=precision,
                            recall=recall, F1=f1, TN=tn, FP=fp, FN=fn, TP=tp))
        scores[name] = prob
    metrics = pd.DataFrame(results)
    metrics.to_csv(out / f"metrics{suffix}.csv", index=False)
    pd.DataFrame(coefficients, columns=["model", "indicator", "coefficient", "odds_ratio"]).to_csv(
        out / f"coefficients{suffix}.csv", index=False)
    if suffix:
        return
    test[KEY + ["fatal"]].assign(**scores).to_csv(out / "test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), layout="constrained")
    for name, color in [("M1", "#006c67"), ("M2", "#b64b23")]:
        precision, recall, _ = precision_recall_curve(y_test, scores[name])
        ap = metrics.set_index("model").loc[name, "AP"]
        ax.step(recall, precision, where="post", label=f"{name}  AP {ap:.4f}", color=color)
    ax.axhline(y_test.mean(), color="#6e7681", linestyle="--", label="Test prevalence")
    ax.set(xlabel="Fatal-class recall", ylabel="Fatal-class precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(out / "precision_recall.png", dpi=180)
    plt.close(fig)


def benchmark(spark, raw, reference, out):
    records = []
    configs = [1, 4]
    for partitions in configs:
        spark.conf.set("spark.sql.shuffle.partitions", partitions)
        joint_summary(eligible(collision_groups(raw.repartition(partitions)))).collect()
    for repeat in range(1, 4):
        for partitions in (configs if repeat % 2 else configs[::-1]):
            spark.conf.set("spark.sql.shuffle.partitions", partitions)
            start = time.perf_counter()
            rows = joint_summary(eligible(collision_groups(raw.repartition(partitions)))).collect()
            elapsed = time.perf_counter() - start
            pd.testing.assert_frame_equal(pd.DataFrame(rows, columns=[*ENV, "n", "fatal"]), reference)
            records.append(dict(partitions=partitions, repeat=repeat, seconds=elapsed,
                                groups=len(rows), collisions=int(reference.n.sum())))
    timings = pd.DataFrame(records)
    timings.to_csv(out / "timing_runs.csv", index=False)
    stat = timings.groupby("partitions").seconds.agg(["median", "min", "max"])
    stat.to_csv(out / "timing_summary.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.4), layout="constrained")
    ax.bar(["1 partition", "4 partitions"], stat["median"], color=["#006c67", "#345777"],
           yerr=[stat["median"]-stat["min"], stat["max"]-stat["median"]], capsize=5, width=.48)
    ax.set(ylabel="Elapsed seconds", xlabel="Input repartition and SQL shuffle setting")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "spark_timing.png", dpi=180)
    plt.close(fig)


def run(path, out):
    out.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.master("local[2]").appName("NCDB 2019")
             .config("spark.ui.enabled", "false").config("spark.sql.adaptive.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        raw = read_source(spark, path)
        groups = collision_groups(raw)
        table = eligible(groups)
        output = joint_summary(table)
        output.show(12)
        joint = output.toPandas()
        joint.to_csv(out / "weather_surface.csv", index=False)
        c = table.orderBy(*KEY).toPandas()
        pandas_source = pd.read_csv(path, dtype=str, keep_default_na=False)
        full_duplicates = raw.count() - raw.dropDuplicates().count()
        assert full_duplicates == pandas_source.duplicated().sum()
        pandas_joint = (pandas_source[KEY + ["C_SEV", *ENV]].drop_duplicates()
                        .assign(fatal=lambda d: d.C_SEV.eq("1").astype(int))
                        .groupby(ENV).fatal.agg(n="size", fatal="sum").reset_index())
        pd.testing.assert_frame_equal(joint, pandas_joint)
        audit = dict(rows=len(pandas_source), fields=len(raw.columns), bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     blanks={col: int(pandas_source[col].eq("").sum()) for col in KEY + FIELDS},
                     exact_duplicates=full_duplicates, keys=groups.count(), collisions=len(c),
                     fatal=int(c.fatal.sum()), python=platform.python_version(), os=platform.platform(),
                     cpu=platform.machine(), available_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                     java=subprocess.run(["java", "-version"], capture_output=True, text=True, check=True).stderr.splitlines()[0],
                     spark=spark.version, pandas=pd.__version__, sklearn=sklearn.__version__)
        (out / "audit.json").write_text(json.dumps(audit, indent=2))
        c.to_csv(out / "collisions.csv", index=False)
        benchmark(spark, raw, joint, out)
    finally:
        spark.stop()
    descriptive(joint, out)
    c["hour_band"] = c.C_HOUR.map({str(h): f"{h//4*4:02d}–{h//4*4+3:02d}" for h in range(24)}).fillna(c.C_HOUR)
    c["vehicle_group"] = c.C_VEHS.map(lambda value: "3+" if value.isdigit() and int(value) >= 3 else value)
    train, test = train_test_split(c, test_size=.2, random_state=42, stratify=c.fatal)
    assert not set(train.C_CASE) & set(test.C_CASE)
    pd.concat([train[KEY].assign(split="train"), test[KEY].assign(split="test")]).to_csv(
        out / "split_assignments.csv", index=False)
    fit_models(train, test, out)
    known = lambda frame: frame[~frame[CONTROL + ENV].isin(SPECIAL).any(axis=1)]
    fit_models(known(train), known(test), out, "_known")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/2019_dataset_en.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.input, args.out)
