# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Objective & Semantic Re-Ranking Engine
# MAGIC
# MAGIC Build a three-stage re-ranking pipeline: score blending, vectorized MMR
# MAGIC diversity selection, and post-hoc creator fairness enforcement, with
# MAGIC results tracked to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Load the candidate dataset
# MAGIC
# MAGIC Run this cell first. The `embedding_vector` column is stored as a
# MAGIC JSON-encoded string in the CSV (a CSV cell can't natively hold a Python
# MAGIC list), so it needs to be parsed back into real numeric values before you
# MAGIC can build a NumPy matrix from it.

# COMMAND ----------

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Path to the candidate file. On Databricks, point this at the uploaded copy,
# e.g. "/Volumes/<catalog>/<schema>/<volume>/candidates_embeddings.csv"
# or "/dbfs/FileStore/candidates_embeddings.csv".
DATA_PATH = "candidates_embeddings.csv"

candidates_df = pd.read_csv(C:\PROJECT)

# Parse the JSON-encoded embedding strings back into real lists of floats
candidates_df["embedding_vector"] = candidates_df["embedding_vector"].apply(json.loads)

# Build a proper NumPy matrix from the parsed embeddings
embeddings_matrix = np.stack(candidates_df["embedding_vector"].values)

# apply_mmr mixes positional indexing (embeddings_matrix) with label indexing
# (candidates_df.loc), so the two must line up: require a clean RangeIndex.
assert candidates_df.index.equals(pd.RangeIndex(len(candidates_df))), "reset_index() before running MMR"
assert embeddings_matrix.dtype.kind == "f", "embeddings must be numeric; check the json.loads parsing step"

print(f"Loaded {len(candidates_df)} candidates")
print(f"Embeddings matrix shape: {embeddings_matrix.shape}")
print(candidates_df[["item_id", "saves_score", "dwell_score", "shares_score", "is_new_creator"]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Execute Composite Score Blending (10 min)
# MAGIC
# MAGIC Combine the separate prediction scores for Saves, Dwell Time, and Private
# MAGIC Shares into a single `base_score`, using the provided enterprise utility
# MAGIC weights.

# COMMAND ----------

CONFIG_WEIGHTS = {
    "saves": 0.5,
    "dwell": 0.3,
    "shares": 0.2
}

def blend_multitask_scores(df, weights):
    """Normalizes and computes weighted sum of multi-task logits."""
    # --- Your code here ---
    # Weighted sum of the three task scores. Dividing by the weight total keeps
    # base_score on the same [0, 1] scale as the inputs even if the config is
    # edited later; with the default weights (sum = 1.0) it changes nothing.
    weight_total = weights["saves"] + weights["dwell"] + weights["shares"]
    df["base_score"] = (
        weights["saves"] * df["saves_score"]
        + weights["dwell"] * df["dwell_score"]
        + weights["shares"] * df["shares_score"]
    ) / weight_total

    return df
    # --- End of your code ---

# COMMAND ----------

# Run Task 1
candidates_df = blend_multitask_scores(candidates_df, CONFIG_WEIGHTS)

# Verify the output
print(candidates_df[["item_id", "base_score"]].head())
print(candidates_df["base_score"].describe())

# Spot-check one row against a hand calculation
row = candidates_df.iloc[0]
expected = 0.5 * row["saves_score"] + 0.3 * row["dwell_score"] + 0.2 * row["shares_score"]
assert np.isclose(row["base_score"], expected)

# COMMAND ----------

# MAGIC %md
# MAGIC **Success Checklist**
# MAGIC - Function correctly calculates the weighted sum of the three score columns.
# MAGIC - A new column named `base_score` is successfully added to the DataFrame.
# MAGIC - The function returns the modified DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Implement the Vectorized MMR Diversity Loop (15 min)
# MAGIC
# MAGIC Select a diverse top-10 list using Maximal Marginal Relevance (MMR). The
# MAGIC similarity computation at each step must be vectorized across all
# MAGIC remaining candidates simultaneously — the outer selection loop itself is
# MAGIC expected and unavoidable in MMR, but there should be no inner
# MAGIC element-wise iteration.

# COMMAND ----------

def apply_mmr(candidates_df, embeddings_matrix, lambda_param=0.5, top_k=10):
    """Selects top_k candidates using a highly optimized, vectorized MMR selection."""
    selected_indices = []
    candidate_indices = list(range(len(candidates_df)))

    # Select first item purely by maximum base relevance
    first_idx = candidates_df['base_score'].idxmax()
    selected_indices.append(first_idx)
    candidate_indices.remove(first_idx)

    while len(selected_indices) < top_k and candidate_indices:
        selected_embeddings = embeddings_matrix[selected_indices]
        remaining_embeddings = embeddings_matrix[candidate_indices]

        # Vectorized similarity matrix calculation
        sim_matrix = cosine_similarity(remaining_embeddings, selected_embeddings)
        max_sims = np.max(sim_matrix, axis=1)

        base_scores = candidates_df.loc[candidate_indices, 'base_score'].values

        # --- Your code here ---
        # One array expression scores every remaining candidate at once:
        # reward relevance, penalize redundancy with the closest already-picked item.
        mmr_scores = (1 - lambda_param) * base_scores - lambda_param * max_sims
        # --- End of your code ---

        best_remaining_idx = np.argmax(mmr_scores)
        best_idx = candidate_indices[best_remaining_idx]

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return candidates_df.loc[selected_indices].copy()

# COMMAND ----------

# Run Task 2
mmr_top10 = apply_mmr(candidates_df, embeddings_matrix, lambda_param=0.5, top_k=10)
print(mmr_top10[["item_id", "base_score", "is_new_creator"]])
assert len(mmr_top10) == 10 and mmr_top10.index.is_unique

# Try lambda_param=0 (pure relevance, no diversity penalty) and
# lambda_param=1 (pure diversity, relevance ignored) to see how the
# selection changes.
mmr_pure_relevance = apply_mmr(candidates_df, embeddings_matrix, lambda_param=0.0, top_k=10)
mmr_pure_diversity = apply_mmr(candidates_df, embeddings_matrix, lambda_param=1.0, top_k=10)

# lambda=0 must reproduce a plain sort by base_score
assert list(mmr_pure_relevance.index) == list(candidates_df["base_score"].nlargest(10).index)

for label, frame in [("lambda=0.0", mmr_pure_relevance), ("lambda=0.5", mmr_top10), ("lambda=1.0", mmr_pure_diversity)]:
    print(f"{label}: mean base_score={frame['base_score'].mean():.4f}, "
          f"items shared with lambda=0 list={len(set(frame.index) & set(mmr_pure_relevance.index))}/10")

# COMMAND ----------

# MAGIC %md
# MAGIC **Success Checklist**
# MAGIC - The MMR formula `(1 - lambda) * relevance - (lambda) * similarity` is correctly implemented.
# MAGIC - The per-step similarity calculation is fully vectorized across all remaining candidates.
# MAGIC - The function returns a DataFrame containing exactly `top_k` items.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Integrate the Creator Parity Filter (13 min)
# MAGIC
# MAGIC Check the MMR-ranked top-10 list. If it contains fewer than 2 new-creator
# MAGIC items, swap out the lowest-scoring established items to meet the 20%
# MAGIC quota.

# COMMAND ----------

def enforce_fairness(selected_df, all_candidates_df, min_new_creators=2):
    """Applies a post-hoc exposure quota override to protect creator equity."""
    new_creator_count = (selected_df['is_new_creator'] == True).sum()

    if new_creator_count >= min_new_creators:
        return selected_df

    # --- Your code here ---
    # 1. Deficit of new creators needed to reach the quota (1, 2, or more).
    deficit = int(min_new_creators - new_creator_count)

    # 2. Best new-creator items not already in the list, highest base_score first.
    unused_new = (
        all_candidates_df[
            (all_candidates_df["is_new_creator"] == True)
            & (~all_candidates_df.index.isin(selected_df.index))
        ]
        .sort_values("base_score", ascending=False)
    )

    # 3. Established items from the tail of the ranked list (lowest-ranked first
    #    out). After MMR the list is ordered by marginal (MMR) score, not raw
    #    base_score, so the tail holds the items with the lowest combined
    #    relevance-plus-novelty value. Dropping by raw base_score instead would
    #    remove the items MMR promoted for diversity and undo stage 2.
    #    Cap the swap count so a small new-creator pool can't raise an
    #    IndexError or change the list length.
    established_in_list = selected_df[selected_df["is_new_creator"] != True]
    n_swaps = min(deficit, len(unused_new), len(established_in_list))

    established_to_drop = established_in_list.tail(n_swaps).index
    new_to_add = unused_new.head(n_swaps)

    # 4. Drop exactly as many established items as new creators added, so the
    #    list length (top_k) is unchanged. Promoted items fill the tail slots.
    final_df = pd.concat([selected_df.drop(index=established_to_drop), new_to_add]).copy()

    assert len(final_df) == len(selected_df), "fairness swap must preserve top_k"
    # --- End of your code ---
    return final_df

# COMMAND ----------

# Test 1: a list that already meets the quota (should return unchanged)
sample_ok = mmr_top10.copy()
sample_ok.loc[sample_ok.index[0], "is_new_creator"] = True
sample_ok.loc[sample_ok.index[1], "is_new_creator"] = True
result_ok = enforce_fairness(sample_ok, candidates_df, min_new_creators=2)
print("Test 1 (already meets quota):", (result_ok["is_new_creator"] == True).sum(), "new creators")
assert result_ok.equals(sample_ok)

# Test 2: the real MMR output, which may have zero new creators
final_list = enforce_fairness(mmr_top10, candidates_df, min_new_creators=2)
print("Test 2 (real MMR output):", (final_list["is_new_creator"] == True).sum(), "new creators")
print(final_list[["item_id", "base_score", "is_new_creator"]])

# Test 3: a manually built list with zero new creators (deficit = 2) must
# receive the two highest-relevance new creators from the full candidate set.
sample_zero = candidates_df[candidates_df["is_new_creator"] == False].nlargest(10, "base_score")
result_zero = enforce_fairness(sample_zero, candidates_df, min_new_creators=2)
top_two_new = candidates_df[candidates_df["is_new_creator"] == True].nlargest(2, "base_score").index
print("Test 3 (zero new creators):", (result_zero["is_new_creator"] == True).sum(), "new creators")
assert len(result_zero) == 10
assert set(top_two_new).issubset(result_zero.index)
assert not set(sample_zero.tail(2).index) & set(result_zero.index)  # lowest-ranked established items dropped

# Test 4: a larger deficit (quota of 3 on the zero-new-creator list)
result_three = enforce_fairness(sample_zero, candidates_df, min_new_creators=3)
assert len(result_three) == 10 and (result_three["is_new_creator"] == True).sum() == 3

# COMMAND ----------

# MAGIC %md
# MAGIC **Success Checklist**
# MAGIC - The function correctly calculates the deficit of new creators.
# MAGIC - The function returns the original DataFrame unchanged if the quota is already met.
# MAGIC - If a deficit exists, the lowest-ranked established items are correctly swapped for the highest-ranked available new creator items.
# MAGIC - The final returned DataFrame always contains `top_k` items.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Programmatically Instrument MLflow Logs (15 min)
# MAGIC
# MAGIC Wrap the full pipeline in an MLflow run, log the `lambda_param`, compute
# MAGIC and log three metrics (NDCG@10, diversity entropy, new creator ratio),
# MAGIC and log a trade-off plot as an artifact.

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def calculate_metrics(final_list, all_candidates_df, embeddings_matrix, all_indices_map, n_clusters=8):
    """Computes precision (NDCG@10), diversity entropy, and new creator exposure ratio."""
    # --- Your code here ---
    k = min(10, len(final_list))

    # 1. NDCG@10 with base_score as the graded relevance (linear gain).
    #    DCG uses final_list's actual order; IDCG uses the top 10 items of the
    #    full pool sorted purely by base_score.
    discounts = 1.0 / np.log2(np.arange(2, k + 2))           # 1/log2(rank + 1)
    actual_gains = final_list["base_score"].to_numpy()[:k]
    ideal_gains = all_candidates_df["base_score"].nlargest(k).to_numpy()
    dcg = float(np.sum(actual_gains * discounts))
    idcg = float(np.sum(ideal_gains * discounts))
    ndcg_10 = dcg / idcg if idcg > 0 else 0.0

    # 2. Diversity entropy: cluster the full pool's embeddings, then measure how
    #    evenly final_list spreads across those clusters. Normalized Shannon
    #    entropy H / ln(n_clusters): 0 = every item in one cluster, 1 = perfectly
    #    even. random_state fixes the clustering so runs are comparable.
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_matrix)
    positions = [all_indices_map[idx] for idx in final_list.index]
    counts = np.bincount(cluster_labels[positions], minlength=n_clusters)
    probs = counts[counts > 0] / counts.sum()
    diversity_entropy = float(-np.sum(probs * np.log(probs)) / np.log(n_clusters))

    # 3. Share of the list that comes from new creators.
    new_creator_ratio = float((final_list["is_new_creator"] == True).mean())

    return {
        "ndcg_10": ndcg_10,
        "diversity_entropy": diversity_entropy,
        "new_creator_ratio": new_creator_ratio,
    }
    # --- End of your code ---

# COMMAND ----------

# --- Your code here ---

# Tracking destination. In a Databricks notebook, runs go to the workspace
# tracking server automatically. To log to the shared Azure ML workspace
# instead, set AZUREML_MLFLOW_URI to the value of
# ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri.
AZUREML_TRACKING_URI = os.environ.get("AZUREML_MLFLOW_URI")
if AZUREML_TRACKING_URI:
    mlflow.set_tracking_uri(AZUREML_TRACKING_URI)

# Optional experiment name (on Databricks this must be a workspace path,
# e.g. "/Users/<you>/reranking-pipeline"). Unset = the notebook's experiment.
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
if EXPERIMENT_NAME:
    mlflow.set_experiment(EXPERIMENT_NAME)

LAMBDA_PARAM = 0.5            # deployed MMR setting
TOP_K = 10
MIN_NEW_CREATORS = 2          # 20% exposure quota on a top-10 list
N_CLUSTERS = 8
LAMBDA_SWEEP = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

def stage_line(name, m):
    return (f"{name:<26} NDCG@10={m['ndcg_10']:.4f}  "
            f"entropy={m['diversity_entropy']:.4f}  new_creator_ratio={m['new_creator_ratio']:.2f}")

with mlflow.start_run(run_name=f"reranking_pipeline_lambda_{LAMBDA_PARAM}") as parent_run:
    # Parameters
    mlflow.log_param("lambda_param", LAMBDA_PARAM)
    mlflow.log_params({
        "top_k": TOP_K,
        "min_new_creators": MIN_NEW_CREATORS,
        "n_clusters": N_CLUSTERS,
        "weight_saves": CONFIG_WEIGHTS["saves"],
        "weight_dwell": CONFIG_WEIGHTS["dwell"],
        "weight_shares": CONFIG_WEIGHTS["shares"],
    })

    # Pipeline: blend -> MMR -> fairness
    blended_df = blend_multitask_scores(candidates_df, CONFIG_WEIGHTS)
    mmr_selected_df = apply_mmr(blended_df, embeddings_matrix, lambda_param=LAMBDA_PARAM, top_k=TOP_K)
    final_ranked_list = enforce_fairness(mmr_selected_df, blended_df, min_new_creators=MIN_NEW_CREATORS)

    all_indices_map = {idx: pos for pos, idx in enumerate(blended_df.index)}
    metrics = calculate_metrics(final_ranked_list, blended_df, embeddings_matrix,
                                all_indices_map, n_clusters=N_CLUSTERS)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # Stage-by-stage view of what each post-processing layer costs, against a
    # relevance-only top-10 with no re-ranking.
    baseline_df = blended_df.nlargest(TOP_K, "base_score")
    stages = {
        "baseline": calculate_metrics(baseline_df, blended_df, embeddings_matrix, all_indices_map, N_CLUSTERS),
        "mmr_only": calculate_metrics(mmr_selected_df, blended_df, embeddings_matrix, all_indices_map, N_CLUSTERS),
        "mmr_fairness": metrics,
    }
    for stage, stage_metrics in stages.items():
        for name, value in stage_metrics.items():
            mlflow.log_metric(f"{stage}_{name}", value)
    mlflow.log_metric("ndcg_cost_vs_baseline", stages["baseline"]["ndcg_10"] - metrics["ndcg_10"])
    mlflow.log_metric("ndcg_cost_of_fairness", stages["mmr_only"]["ndcg_10"] - metrics["ndcg_10"])

    print("Deployed pipeline (lambda = %.1f)" % LAMBDA_PARAM)
    for stage, stage_metrics in stages.items():
        print(stage_line(stage, stage_metrics))

    # Lambda sweep as nested runs: each child logs its own lambda_param and the
    # three metrics, before and after the fairness layer.
    sweep_rows = []
    for lam in LAMBDA_SWEEP:
        with mlflow.start_run(run_name=f"lambda_{lam:.1f}", nested=True):
            mlflow.log_param("lambda_param", lam)
            mmr_l = apply_mmr(blended_df, embeddings_matrix, lambda_param=lam, top_k=TOP_K)
            fair_l = enforce_fairness(mmr_l, blended_df, min_new_creators=MIN_NEW_CREATORS)
            pre = calculate_metrics(mmr_l, blended_df, embeddings_matrix, all_indices_map, N_CLUSTERS)
            post = calculate_metrics(fair_l, blended_df, embeddings_matrix, all_indices_map, N_CLUSTERS)
            for name, value in post.items():
                mlflow.log_metric(name, value)
            for name, value in pre.items():
                mlflow.log_metric(f"pre_fairness_{name}", value)
            sweep_rows.append({"lambda_param": lam,
                               **{f"pre_fairness_{k}": v for k, v in pre.items()},
                               **post})

    sweep_df = pd.DataFrame(sweep_rows)
    print("\nLambda sweep (after fairness):")
    print(sweep_df[["lambda_param", "ndcg_10", "diversity_entropy", "new_creator_ratio"]].to_string(index=False))

    # Trade-off plot, three panels:
    #   (a) NDCG@10 vs. diversity entropy for each stage at the deployed lambda
    #   (b) NDCG@10 across the lambda sweep, before and after the fairness layer
    #   (c) diversity entropy across the same sweep
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    C_MMR, C_FAIR = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) stage path at the deployed lambda
    ax = axes[0]
    path = [("Baseline (relevance only)", stages["baseline"], INK, "*", 260),
            (f"MMR only (λ={LAMBDA_PARAM})", stages["mmr_only"], C_MMR, "o", 110),
            ("MMR + creator fairness", stages["mmr_fairness"], C_FAIR, "s", 110)]
    for (_, a, *_), (_, b, *_) in zip(path, path[1:]):
        ax.annotate("", xy=(b["diversity_entropy"], b["ndcg_10"]),
                    xytext=(a["diversity_entropy"], a["ndcg_10"]),
                    arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.2, shrinkA=9, shrinkB=9))
    offsets = [(10, -10), (10, 6), (10, -30)]
    aligns = ["left", "left", "right"]
    for (label, m, color, marker, size), off, ha in zip(path, offsets, aligns):
        ax.scatter(m["diversity_entropy"], m["ndcg_10"], marker=marker, s=size, color=color,
                   edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.annotate(f"{label}\nNDCG {m['ndcg_10']:.3f} · H {m['diversity_entropy']:.3f} · "
                    f"new {m['new_creator_ratio']:.0%}",
                    (m["diversity_entropy"], m["ndcg_10"]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8.5, color=INK)
    ax.set_xlim(0.79, 0.94)
    ax.set_ylim(0.968, 1.004)
    ax.set(xlabel="Diversity entropy (normalized, 8 clusters)", ylabel="NDCG@10")
    ax.set_title(f"(a) Cost of each layer at λ = {LAMBDA_PARAM}", loc="left", fontsize=11, color=INK)

    # (b) and (c) lambda sweep
    for ax, key, title in [(axes[1], "ndcg_10", "(b) NDCG@10 across the λ sweep"),
                           (axes[2], "diversity_entropy", "(c) Diversity entropy across the λ sweep")]:
        ax.axhline(stages["baseline"][key], color=INK_2, linestyle="--", linewidth=1)
        ax.axvline(LAMBDA_PARAM, color=INK_2, linestyle=":", linewidth=1)
        ax.plot(sweep_df["lambda_param"], sweep_df[f"pre_fairness_{key}"], color=C_MMR, linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label="MMR only")
        ax.plot(sweep_df["lambda_param"], sweep_df[key], color=C_FAIR, linewidth=2,
                marker="s", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="MMR + creator fairness")
        ax.set(xlabel="MMR λ (0 = pure relevance, 1 = pure diversity)", xticks=np.arange(0, 1.01, 0.2))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[1].set_ylabel("NDCG@10")
    axes[2].set_ylabel("Diversity entropy")
    axes[1].annotate("baseline", (1.0, stages["baseline"]["ndcg_10"]), textcoords="offset points",
                     xytext=(0, -12), ha="right", fontsize=8.5, color=INK_2)
    axes[2].annotate("baseline", (0.0, stages["baseline"]["diversity_entropy"]), textcoords="offset points",
                     xytext=(2, 5), ha="left", fontsize=8.5, color=INK_2)
    axes[1].annotate(f"deployed λ={LAMBDA_PARAM}", (LAMBDA_PARAM, axes[1].get_ylim()[0]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8.5, color=INK_2)
    axes[1].legend(frameon=False, loc="lower left", labelcolor=INK)

    fig.tight_layout()
    fig.savefig("tradeoff_plot.png", dpi=150, facecolor=fig.get_facecolor())
    mlflow.log_artifact("tradeoff_plot.png")
    plt.close(fig)

    # Supporting artifacts: the served list and the sweep table
    final_ranked_list.drop(columns=["embedding_vector"]).to_csv("final_ranked_list.csv", index=False)
    sweep_df.to_csv("lambda_sweep_metrics.csv", index=False)
    mlflow.log_artifact("final_ranked_list.csv")
    mlflow.log_artifact("lambda_sweep_metrics.csv")

    print(f"\nMLflow run ID: {parent_run.info.run_id}")

print("\nFinal ranked list (served order):")
print(final_ranked_list[["item_id", "base_score", "is_new_creator"]].to_string())

# --- End of your code ---

# COMMAND ----------

# MAGIC %md
# MAGIC **Success Checklist**
# MAGIC - The entire pipeline execution is wrapped within an `mlflow.start_run()` context.
# MAGIC - The MMR `lambda_param` is logged as a parameter.
# MAGIC - Three distinct metrics (NDCG@10, Diversity Entropy, New Creator Ratio) are logged.
# MAGIC - At least one artifact (e.g., a trade-off plot image file) is logged to the run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Common Mistakes to Avoid
# MAGIC
# MAGIC - **Inefficient loops**: calculating MMR scores one-by-one instead of vectorizing across all remaining candidates in a single expression per step.
# MAGIC - **Incorrect indexing**: forgetting to remove selected indices from the candidate pool in the MMR loop, causing infinite loops or repeated selections.
# MAGIC - **Stateful DataFrame modification**: modifying a DataFrame without returning it, or relying on global state — each function should accept a DataFrame and return one.
# MAGIC - **Authentication errors**: forgetting to set the Azure ML tracking URI correctly, causing MLflow to log locally instead of to the shared workspace registry.
# MAGIC - **Embedding parsing**: forgetting to parse the `embedding_vector` column with `json.loads()` before calling `np.stack()` — skipping this produces a silently wrong array of strings rather than a numeric matrix.
