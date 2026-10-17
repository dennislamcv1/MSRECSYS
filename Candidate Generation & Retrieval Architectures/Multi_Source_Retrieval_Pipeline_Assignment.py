# Databricks notebook source
# MAGIC %md
# MAGIC # (APPLY/ASSESS) GRADED ASSIGNMENT (PROGRAMMING) | Multi-Source Retrieval Pipeline (45 min)
# MAGIC
# MAGIC **Assignment type:** Graded Assignment (Programming)
# MAGIC **Tools:** Azure Databricks cluster
# MAGIC
# MAGIC ## Overview
# MAGIC You will build and evaluate a production-grade, multi-source retrieval pipeline within an Azure Databricks notebook. This hands-on project requires you to orchestrate three distinct candidate generation models (ALS, Two-Tower, and LightGCN), blend their outputs, and evaluate whether the combined system provides a recall benefit compared to any single source.
# MAGIC
# MAGIC ## Goals
# MAGIC By completing this activity, you will:
# MAGIC - Construct an integrated multi-source retrieval system blending candidates from collaborative filtering, deep neural, and graph-based models.
# MAGIC - Implement parallel candidate generation using Python's `concurrent.futures`, applied only where it's safe to run concurrently.
# MAGIC - Apply global string ID deduplication to create a unified, clean candidate pool.
# MAGIC - Evaluate a combined candidate pool using the Recall@100 metric and compare it against each individual source.
# MAGIC
# MAGIC ## Scenario
# MAGIC You are a Lead Machine Learning Engineer at a large enterprise platform, tasked with improving the quality and diversity of your recommendation engine's retrieval stage. Your existing retrieval model fails to capture niche user interests and struggles with the cold-start problem. Your objective is to architect a new solution that combines the strengths of three different models: an ALS collaborative filtering model, a Two-Tower neural network, and a LightGCN graph-based model.
# MAGIC
# MAGIC A latency-conscious design is part of the engineering brief for this pipeline — production retrieval systems are typically expected to respond quickly, which is why this task emphasizes parallel orchestration wherever it's safe to run concurrently. Treat any specific latency target as a design goal to discuss in your architecture document, not a number this notebook measures or enforces.
# MAGIC
# MAGIC Your task is to write a master Python script that generates candidates from all three sources — running the non-Spark sources in parallel — merges their outputs into a single deduplicated pool, and evaluates the final combined candidate list against a Recall@100 benchmark.
# MAGIC
# MAGIC ## Solution summary
# MAGIC - **Task 1** loads the five artifacts from the Unity Catalog Volume, inspects them, and asserts shape and schema before anything downstream runs.
# MAGIC - **Task 2** scores ALS from its raw factor matrices on the main thread, then runs Two-Tower and LightGCN concurrently in a `ThreadPoolExecutor(max_workers=2)`.
# MAGIC - **Task 3** normalizes every item ID to one canonical string form, merges all three sources with a single global set union, and reports Recall@100 per source against the combined pool (scored in full, because a set has no rank order). A supplementary cell adds like-for-like comparisons, and the last code cell prints the final verdict.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 1: Initialize the Orchestration Environment (10 min)
# MAGIC
# MAGIC **Context:** Before you can orchestrate the models, you must load all necessary libraries and pre-computed artifacts into your Azure Databricks notebook. These artifacts are stored in a **Unity Catalog Volume**, not the legacy Databricks File System (DBFS) — Volumes are required here because Spark model artifacts like the ALS model involve distributed reads/writes across executor tasks, which the legacy DBFS/Workspace filesystem does not support.
# MAGIC
# MAGIC **Practice:**
# MAGIC - Complete the code block below by loading each artifact from its Volume path into a variable.
# MAGIC - After loading each artifact, use a new cell to inspect it (`.shape`, `.head()`, `spark.version`).
# MAGIC - Run a local validation check on the loaded artifacts before moving on — confirm dimensions and expected columns match what you expect.
# MAGIC
# MAGIC **Consider:** Why is it important to verify the dimensions of the loaded `two_tower_embeddings` array before proceeding? What kind of error could a dimension mismatch cause in the next task?
# MAGIC
# MAGIC **Success Checklist:**
# MAGIC - [x] The ALS model is successfully loaded into a variable using `ALSModel.load()`.
# MAGIC - [x] The Two-Tower embeddings and item IDs are loaded into NumPy arrays.
# MAGIC - [x] The LightGCN candidates and ground truth data are loaded into pandas DataFrames.
# MAGIC - [x] A local validation cell confirms array/DataFrame shapes and schema before proceeding.
# MAGIC - [x] All notebook cells execute without file-not-found or deserialization errors.

# COMMAND ----------

import concurrent.futures
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Databricks already provides `spark`; getOrCreate() simply returns that same session.
spark = SparkSession.builder.getOrCreate()

# Paths to pre-computed artifacts in a Unity Catalog Volume
CATALOG_NAME = "courses_workspace"
SCHEMA_NAME = "default"
VOLUME_NAME = "lab_assets_vol"
ASSETS_DIR = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"

ALS_MODEL_PATH = f"{ASSETS_DIR}/als_model"
TWO_TOWER_EMBEDDINGS_PATH = f"{ASSETS_DIR}/two_tower_item_embeddings.npy"
TWO_TOWER_IDS_PATH = f"{ASSETS_DIR}/two_tower_item_ids.npy"
LIGHTGCN_CANDIDATES_PATH = f"{ASSETS_DIR}/lightgcn_candidates.parquet"
GROUND_TRUTH_PATH = f"{ASSETS_DIR}/ground_truth_interactions.parquet"

# Load the ALS model (a Spark ML model directory, read by Spark from the Volume)
als_model = ALSModel.load(ALS_MODEL_PATH)

# Load the Two-Tower embeddings and IDs.
# Row i of the embedding matrix belongs to the item at two_tower_item_ids[i].
# allow_pickle=True also covers IDs that were saved as a NumPy object array.
two_tower_embeddings = np.load(TWO_TOWER_EMBEDDINGS_PATH)
two_tower_item_ids = np.load(TWO_TOWER_IDS_PATH, allow_pickle=True)

# Load the LightGCN candidates
lightgcn_candidates = pd.read_parquet(LIGHTGCN_CANDIDATES_PATH)

# Load the ground truth evaluation data
ground_truth = pd.read_parquet(GROUND_TRUTH_PATH)

# COMMAND ----------

# Inspect the loaded artifacts
print("Spark version:", spark.version)
print("ALS model loaded as:", type(als_model).__name__)

print("\nTwo-Tower embeddings shape:", two_tower_embeddings.shape, "| dtype:", two_tower_embeddings.dtype)
print("Two-Tower item IDs:", len(two_tower_item_ids), "| dtype:", two_tower_item_ids.dtype,
      "| sample:", list(two_tower_item_ids[:3]))

print("\nLightGCN candidates shape:", lightgcn_candidates.shape)
print(lightgcn_candidates.dtypes)
print(lightgcn_candidates.head())

print("\nGround truth shape:", ground_truth.shape)
print(ground_truth.dtypes)
print(ground_truth.head())

# COMMAND ----------

# Local validation checks — confirm dimensions and schema before moving on
assert als_model is not None, "ALS model failed to load"

assert two_tower_embeddings.ndim == 2, (
    f"Expected a 2-D (n_items x embedding_dim) matrix, got shape {two_tower_embeddings.shape}"
)
assert two_tower_embeddings.shape[0] == len(two_tower_item_ids), "Embedding/ID count mismatch"

assert "user_id" in lightgcn_candidates.columns and "item_id" in lightgcn_candidates.columns, (
    "LightGCN candidates are missing user_id / item_id columns"
)
assert "user_id" in ground_truth.columns and "item_id" in ground_truth.columns, (
    "Ground truth is missing user_id / item_id columns"
)
assert len(ground_truth) > 0, "Ground truth is empty — Recall cannot be computed"

# If every assert above passes, execution reaches this line without error.
print("All validation checks passed.")
print(f"  Two-Tower: {two_tower_embeddings.shape[0]} items x {two_tower_embeddings.shape[1]} dims")
print(f"  LightGCN:  {len(lightgcn_candidates)} rows for {lightgcn_candidates['user_id'].nunique()} users")
print(f"  Ground truth: {len(ground_truth)} interactions for {ground_truth['user_id'].nunique()} users")

# COMMAND ----------

# MAGIC %md
# MAGIC **Consider — why verify the `two_tower_embeddings` dimensions?**
# MAGIC Row *i* of the embedding matrix must describe the item at `two_tower_item_ids[i]`. If the row count and the ID count differ, `two_tower_item_ids[top_indices]` in Task 2 either raises an `IndexError` (an index past the end of the ID array) or, worse, runs cleanly and maps every score to the wrong item, which quietly corrupts the recall numbers. The second dimension matters too: `np.dot(user_embeddings, two_tower_embeddings.T)` raises `ValueError: shapes not aligned` if the user and item vectors have different widths. A matrix that isn't 2-D at all would break `.shape[1]`, which Task 2 uses to size the user embeddings. Checking up front turns all of these into one clear failure at load time.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 2: Implement Candidate Generation (15 min)
# MAGIC
# MAGIC **Context:** You'll generate candidates from all three sources. The ALS lookup depends on the active SparkContext, and Spark operations are **not safe to call from inside a `ThreadPoolExecutor` worker thread** under Databricks' pinned-thread execution model. So ALS runs on the main thread, and only the two NumPy/pandas-based sources — Two-Tower and LightGCN — run concurrently. This still gives you a real parallelization benefit without the cross-thread Spark issue.
# MAGIC
# MAGIC **Practice:**
# MAGIC - Complete `generate_als_candidates` by ranking each user's candidate items via a dot product between their factor vector and every item's factor vector, then taking the top-k.
# MAGIC - Call `generate_als_candidates` directly (main thread) before entering the executor block.
# MAGIC - Instantiate a `concurrent.futures.ThreadPoolExecutor` with `max_workers=2`.
# MAGIC - Within the `with` block, submit `generate_two_tower_candidates` and `generate_lightgcn_candidates` to the executor.
# MAGIC - Collect each function's output into a separate results variable (`als_results`, `tt_results`, `lgcn_results`).
# MAGIC
# MAGIC **Consider:** Why does ALS need to stay on the main thread here, while Two-Tower and LightGCN don't? What's different about what each function actually does under the hood?
# MAGIC
# MAGIC **Success Checklist:**
# MAGIC - [x] `generate_als_candidates` produces ranked candidates from the model's raw factors, without calling `recommendForUserSubset`.
# MAGIC - [x] `ThreadPoolExecutor` is used to orchestrate the two non-Spark generation functions.
# MAGIC - [x] The `submit` method is called for each of those two functions.
# MAGIC - [x] The `.result()` method is called for each future to retrieve its output.
# MAGIC - [x] The code block executes without errors and produces candidate lists from all three sources.

# COMMAND ----------

# Sample user inputs
# NOTE: user_id is an INTEGER throughout this pipeline; item_id is a STRING ("item_N").
# Keeping this consistent avoids type-mismatch bugs when merging candidates later.
user_ids = ground_truth["user_id"].unique().tolist()
embedding_dim = two_tower_embeddings.shape[1]
rng = np.random.default_rng(seed=42)
user_embeddings = rng.normal(size=(len(user_ids), embedding_dim))

# COMMAND ----------

# --- Starter functions for each generation source ---

def _top_k_desc(scores, k):
    """Indices of the k highest scores, highest first.

    argpartition finds the top k in O(n_items); only those k are then sorted.
    """
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]


def generate_als_candidates(user_ids, top_k=100):
    als_model = ALSModel.load(ALS_MODEL_PATH)

    # Read the model's learned factors directly rather than calling
    # als_model.recommendForUserSubset(), which relies on Spark
    # higher-order functions not supported on this cluster type.
    user_factors_df = als_model.userFactors.toPandas()
    item_factors_df = als_model.itemFactors.toPandas()

    # 1. Stack the factor vectors into dense NumPy matrices (rows x rank)
    user_matrix = np.array(user_factors_df["features"].tolist(), dtype=np.float64)
    item_matrix = np.array(item_factors_df["features"].tolist(), dtype=np.float64)

    # Spark does not guarantee the row order of itemFactors, so each matrix row is mapped
    # back to the ALS item id it belongs to rather than assuming row position == item id.
    item_ids_by_row = item_factors_df["id"].to_numpy()

    # 2. user_id -> row index lookup (ALS ids are integers, matching our integer user_id)
    user_row = {int(uid): row for row, uid in enumerate(user_factors_df["id"])}

    # 3. Score every item for each requested user and keep the top_k, highest first
    rows = []
    for uid in user_ids:
        row = user_row.get(int(uid))
        if row is None:
            # Cold-start user: no learned ALS factors, so ALS contributes nothing for them.
            continue
        scores = item_matrix @ user_matrix[row]
        for item_idx in _top_k_desc(scores, top_k):
            rows.append({"user_id": uid, "item_id": f"item_{item_ids_by_row[item_idx]}"})

    # 4. Return a pandas DataFrame of the collected rows
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def generate_two_tower_candidates(user_embeddings, top_k=100):
    scores = np.dot(user_embeddings, two_tower_embeddings.T)
    # Sort descending so the top similarity scores come first
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
    return two_tower_item_ids[top_indices]


def generate_lightgcn_candidates(user_ids, top_k=100):
    return lightgcn_candidates[lightgcn_candidates["user_id"].isin(user_ids)
    ].groupby("user_id").head(top_k)

# COMMAND ----------

TOP_K = 100

# 1. Run generate_als_candidates on the MAIN THREAD (not inside the executor).
#    It loads a Spark model and calls .toPandas(), both of which submit Spark jobs.
t0 = time.perf_counter()
als_results = generate_als_candidates(user_ids, top_k=TOP_K)
als_seconds = time.perf_counter() - t0

# 2. Use a ThreadPoolExecutor (max_workers=2) to run Two-Tower and LightGCN concurrently.
#    Both are pure NumPy/pandas work that only reads in-memory data, so they are thread-safe.
t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    tt_future = executor.submit(generate_two_tower_candidates, user_embeddings, TOP_K)
    lgcn_future = executor.submit(generate_lightgcn_candidates, user_ids, TOP_K)

    # .result() blocks until each future finishes and re-raises any worker exception here.
    tt_results = tt_future.result()
    lgcn_results = lgcn_future.result()
parallel_seconds = time.perf_counter() - t0

print(f"ALS returned {len(als_results)} recommendations.")
print(f"Two-Tower returned candidates for {len(tt_results)} users.")
print(f"LightGCN returned {len(lgcn_results)} candidate rows.")

print(f"\nALS covered {als_results['user_id'].nunique()} of {len(user_ids)} users "
      f"(the rest have no learned ALS factors, i.e. cold start).")
print(f"Stage timings (informational only, not a latency target): "
      f"ALS on main thread {als_seconds:.2f}s | Two-Tower + LightGCN in parallel {parallel_seconds:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC **Consider — why does ALS stay on the main thread while the other two don't?**
# MAGIC `generate_als_candidates` is not just arithmetic. `ALSModel.load()` and the two `.toPandas()` calls go through the SparkSession and the JVM (via Py4J) and submit Spark jobs to the cluster. Under Databricks' pinned-thread model each Python thread is paired with its own JVM thread, and Spark's per-thread state (job group, scheduler pool and other local properties) lives with the thread that owns the session; a fresh pool worker doesn't inherit it, so Spark calls from there are unsafe. Once the factors are pulled to the driver, the scoring itself is plain NumPy.
# MAGIC
# MAGIC Two-Tower is a NumPy matrix multiply plus an `argsort`; LightGCN is a pandas filter plus `groupby().head()`. Neither touches Spark, and both only read arrays already in driver memory, so running them side by side can't corrupt shared state. NumPy's matrix multiply also releases Python's GIL, so the two really do overlap: the parallel stage takes roughly as long as the slower of the two rather than the sum.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 3: Blend, Deduplicate, and Evaluate (20 min)
# MAGIC
# MAGIC **Context:** With candidates from all three sources, your final task is to merge them into a single pool, handle duplicates, and evaluate the result. The most critical steps are ensuring all item identifiers are consistent string types and that deduplication happens globally, across all three sources at once.
# MAGIC
# MAGIC **Practice:**
# MAGIC - Convert the item IDs from each of the three results into a Python list of strings.
# MAGIC - Use the `|` (union) operator to combine the three sets into one final deduplicated set.
# MAGIC - **Note:** the per-source lists preserve rank order, but a Python `set` does not — when evaluating the combined pool, decide whether truncating to top-100 is meaningful for an unordered set, or whether the combined pool should be evaluated in full.
# MAGIC - Complete the `recall_at_k` function: intersection of candidate IDs and ground truth IDs, divided by the total number of ground truth items.
# MAGIC - Print Recall@100 for ALS, Two-Tower, LightGCN, and the combined pool.
# MAGIC - Analyze your output: Does the combined Recall show a lift over the best individual source? If not, what could be wrong — or is the combined pool being truncated in a way that doesn't reflect its actual contents?
# MAGIC
# MAGIC **Success Checklist:**
# MAGIC - [x] All item IDs are correctly cast to strings before being added to sets.
# MAGIC - [x] A set union is used to perform a global deduplication across all three sources.
# MAGIC - [x] The `recall_at_k` function correctly calculates the metric.
# MAGIC - [x] You report and briefly justify the Combined Recall result — a numeric lift over every individual source is the expected typical outcome with this lab's data, but your writeup should explain your result rather than assume it.

# COMMAND ----------

def to_item_id_str(raw_id):
    """Return the canonical string form of one item ID, or None for a null.

    Decodes bytes (np.load can return b'item_7'), casts everything else to str, and maps
    bare integers onto the shared "item_N" convention so that 7, "7" and "item_7" all
    deduplicate to the same key.
    """
    if isinstance(raw_id, (bytes, np.bytes_)):
        raw_id = raw_id.decode("utf-8")
    elif pd.isna(raw_id):
        return None
    item_id = str(raw_id).strip()
    if item_id.isdigit():
        item_id = f"item_{item_id}"
    return item_id


def to_str_ids(values):
    """Normalize an iterable of raw item IDs into a list of canonical strings, dropping nulls."""
    return [s for s in (to_item_id_str(v) for v in values) if s is not None]


# 1. Normalize all candidate item IDs from each source to a consistent string format.
#    Each list keeps rank order within a user (users appear one after another).
als_ids = to_str_ids(als_results["item_id"])
tt_ids = to_str_ids(np.asarray(tt_results).ravel())   # row-major: user 0's top-100, then user 1's, ...
lgcn_ids = to_str_ids(lgcn_results["item_id"])

# 2. Use the set union operator to merge the three sets into a single, deduplicated set.
#    One union across all sources (and all users) = global deduplication.
combined_candidate_ids = set(als_ids) | set(tt_ids) | set(lgcn_ids)
combined_candidate_list = list(combined_candidate_ids)


# 3. Implement the recall@k function
def recall_at_k(candidate_ids, ground_truth_ids_set, k=100):
    """|candidates[:k] ∩ ground truth| / |ground truth|.  k=None scores the full candidate list."""
    if not ground_truth_ids_set:
        return 0.0
    candidates = list(candidate_ids)
    if k is not None:
        candidates = candidates[:k]
    hits = len(set(candidates) & ground_truth_ids_set)
    return hits / len(ground_truth_ids_set)


# 4. Calculate and print Recall@100 for each source AND the combined pool
gt_set = set(to_str_ids(ground_truth["item_id"]))

als_recall = recall_at_k(als_ids, gt_set, k=100)
tt_recall = recall_at_k(tt_ids, gt_set, k=100)
lgcn_recall = recall_at_k(lgcn_ids, gt_set, k=100)
# The combined pool is a set with no rank order, so slicing it at [:100] would keep an arbitrary
# 100 items. It is scored in full instead.
combined_recall = recall_at_k(combined_candidate_list, gt_set, k=None)

total_rows = len(als_ids) + len(tt_ids) + len(lgcn_ids)
distinct_per_source = len(set(als_ids)) + len(set(tt_ids)) + len(set(lgcn_ids))
print(f"Candidate rows: ALS {len(als_ids)} | Two-Tower {len(tt_ids)} | LightGCN {len(lgcn_ids)} | total {total_rows}")
print(f"Distinct items per source: ALS {len(set(als_ids))} | Two-Tower {len(set(tt_ids))} | LightGCN {len(set(lgcn_ids))}")
print(f"Combined deduplicated pool: {len(combined_candidate_ids)} unique items "
      f"({distinct_per_source - len(combined_candidate_ids)} cross-source duplicates removed)")
print(f"Ground truth: {len(gt_set)} distinct items\n")

print(f"ALS Recall@100: {als_recall:.4f}")
print(f"Two-Tower Recall@100: {tt_recall:.4f}")
print(f"LightGCN Recall@100: {lgcn_recall:.4f}")
print(f"Combined Recall (full deduplicated pool): {combined_recall:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Supplementary: like-for-like comparisons
# MAGIC The required metric above compares each source's first 100 list entries with the combined pool scored in full. Because each per-source list is flattened user by user, its first 100 entries are essentially one user's top-100, measured against every user's ground truth, while the combined pool spans all users and all three sources. That makes the required comparison lopsided in the combined pool's favour. Two extra views put the sources on the same footing:
# MAGIC - **Global coverage (full pool):** every source scored on all of its candidates (`k=None`), the same basis as the combined pool.
# MAGIC - **Mean per-user Recall@100:** for each ground-truth user, that user's own top-100 from a source against that user's own ground truth, averaged over users. For the combined row, each user's pool is the union of their three top-100 lists (up to 300 items).
# MAGIC
# MAGIC The last column counts ground-truth items that only one source retrieved, which is what each source uniquely adds to the blend. The cell prints a Markdown table that can be pasted straight into the architecture document.

# COMMAND ----------

# (a) Global coverage: each source's FULL candidate pool, same k=None basis as the combined pool
als_full = recall_at_k(als_ids, gt_set, k=None)
tt_full = recall_at_k(tt_ids, gt_set, k=None)
lgcn_full = recall_at_k(lgcn_ids, gt_set, k=None)

# (b) Mean per-user Recall@100
gt_by_user = {}
for uid, iid in zip(ground_truth["user_id"], ground_truth["item_id"]):
    s = to_item_id_str(iid)
    if s is not None:
        gt_by_user.setdefault(uid, set()).add(s)


def lists_by_user(df):
    """{user_id: [item ids in rank order]} for a long-format candidate DataFrame."""
    out = {}
    for uid, iid in zip(df["user_id"], df["item_id"]):
        s = to_item_id_str(iid)
        if s is not None:
            out.setdefault(uid, []).append(s)
    return out


als_by_user = lists_by_user(als_results)
lgcn_by_user = lists_by_user(lgcn_results)
tt_by_user = {uid: to_str_ids(row) for uid, row in zip(user_ids, np.asarray(tt_results))}


def mean_user_recall(candidates_by_user, k=TOP_K):
    """Average over ground-truth users; a user with no candidates from a source scores 0."""
    scores = []
    for uid, gt_items in gt_by_user.items():
        cands = candidates_by_user.get(uid, [])
        if k is not None:
            cands = cands[:k]
        scores.append(len(set(cands) & gt_items) / len(gt_items))
    return float(np.mean(scores)) if scores else 0.0


combined_by_user = {
    uid: list(set(als_by_user.get(uid, [])[:TOP_K])
              | set(tt_by_user.get(uid, [])[:TOP_K])
              | set(lgcn_by_user.get(uid, [])[:TOP_K]))
    for uid in gt_by_user
}

als_user = mean_user_recall(als_by_user)
tt_user = mean_user_recall(tt_by_user)
lgcn_user = mean_user_recall(lgcn_by_user)
combined_user = mean_user_recall(combined_by_user, k=None)
avg_pool = float(np.mean([len(p) for p in combined_by_user.values()])) if combined_by_user else 0.0

# (c) Ground-truth items that only one source retrieved
als_hits, tt_hits, lgcn_hits = set(als_ids) & gt_set, set(tt_ids) & gt_set, set(lgcn_ids) & gt_set
als_only = als_hits - tt_hits - lgcn_hits
tt_only = tt_hits - als_hits - lgcn_hits
lgcn_only = lgcn_hits - als_hits - tt_hits

print("| Source | Recall@100 (required metric) | Global coverage, full pool | Mean per-user Recall@100 | GT items only this source found |")
print("|---|---|---|---|---|")
for name, r100, full, user_r, only in [
    ("ALS", als_recall, als_full, als_user, len(als_only)),
    ("Two-Tower", tt_recall, tt_full, tt_user, len(tt_only)),
    ("LightGCN", lgcn_recall, lgcn_full, lgcn_user, len(lgcn_only)),
]:
    print(f"| {name} | {r100:.4f} | {full:.4f} | {user_r:.4f} | {only} |")
print(f"| **Combined pool** | {combined_recall:.4f} (full pool) | {combined_recall:.4f} | "
      f"{combined_user:.4f} (per-user pool, avg {avg_pool:.0f} items) | n/a |")

# COMMAND ----------

# Final verdict: lift of the combined pool over the best individual source
required = {"ALS": als_recall, "Two-Tower": tt_recall, "LightGCN": lgcn_recall}
like_for_like = {"ALS": als_full, "Two-Tower": tt_full, "LightGCN": lgcn_full}
best_req = max(required, key=required.get)
best_full = max(like_for_like, key=like_for_like.get)

print("Evaluation basis: per-source lists scored at their first 100 ranked entries (as specified); "
      "combined pool scored in FULL (k=None) because a set has no rank order.")
print(f"Lift vs best source, required metric:    {combined_recall - required[best_req]:+.4f} "
      f"(best: {best_req}, {required[best_req]:.4f})")
print(f"Lift vs best source, like-for-like pools: {combined_recall - like_for_like[best_full]:+.4f} "
      f"(best: {best_full}, {like_for_like[best_full]:.4f})")
print(f"Combined Recall (full deduplicated pool): {combined_recall:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis of the result
# MAGIC - **Evaluation choice.** The combined pool was scored in full (`k=None`), not sliced at `[:100]`. A Python `set` has no order, so any 100 items taken from it would be an arbitrary sample; slicing it would understate the pool and make the result depend on hash ordering.
# MAGIC - **Is there a lift?** On this metric the combined pool can't score below any single source: it is the union of every source's candidates, so it contains every hit any source made. The lift is therefore zero or positive by construction. Whether it is strictly positive depends on the "only this source found" column — if each source retrieves at least some relevant items the others miss, the blend shows a real gain, and that is the evidence the three models are complementary rather than redundant.
# MAGIC - **Reading the size of the lift.** The required comparison flatters the combined pool, because each per-source score covers about one user's top-100 while the pool covers every user and every source. The like-for-like row (each source's full pool) is the fairer measure of what blending adds, and the per-user column shows whether the gain holds up for individual users.
# MAGIC - **Watch for saturation.** Global coverage is pooled across all users, so once (users × 100) is large next to the catalog, every source's full pool already touches almost every item and all the global numbers crowd toward 1.0. In that regime the global lift says little, and the mean per-user column is the meaningful measure of what blending adds for an individual user.
# MAGIC - **Lab caveats.** Two-Tower user embeddings are seeded random vectors here, so that source behaves like a random-query baseline: what it adds is coverage, not learned relevance. ALS skips users without learned factors (cold start), and LightGCN's `head(top_k)` assumes the parquet is already ranked within each user.
# MAGIC - **If no lift appeared**, the usual causes are mismatched ID formats across sources (e.g. `123` vs `"item_123"`, or bytes vs str), deduplicating inside each source instead of globally, or truncating the unordered combined set.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common Mistakes to Avoid
# MAGIC - **Running ALS from a worker thread**: calling `generate_als_candidates` from inside the `ThreadPoolExecutor` alongside the other two functions will fail under Databricks' pinned-thread execution model. Keep it on the main thread.
# MAGIC - **Using `recommendForUserSubset` on a Unity Catalog Shared-access-mode cluster**: this method depends on Spark higher-order functions that this cluster type does not support, and will fail with a `UC_COMMAND_NOT_SUPPORTED` error.
# MAGIC - **Inconsistent ID Types**: forgetting to cast all item identifiers to a string format before deduplication. A set will treat the integer `123` and the string `'123'` as two distinct items.
# MAGIC - **Source-Level Deduplication**: deduplicating within each source's candidate list before merging misses cross-source duplicates.
# MAGIC - **Incorrect Recall Calculation**: calculating recall based on raw hit count without normalizing by the ground truth set size.
# MAGIC - **Truncating an unordered set at `[:100]`**: slicing the combined `set` after union does not give you a meaningful "top 100," since sets have no defined order.
# MAGIC
# MAGIC ### Key Points
# MAGIC - Multi-source blending is a production-standard technique for increasing recall and ensuring coverage across diverse user segments and item types.
# MAGIC - Parallelization should be applied where it's safe — I/O-bound, non-Spark operations are good candidates for a thread pool; Spark-dependent operations are not safe to run from arbitrary worker threads under Databricks' pinned-thread execution model.
# MAGIC - Global string ID deduplication is the required method for creating a clean, unified candidate pool from multiple sources.
