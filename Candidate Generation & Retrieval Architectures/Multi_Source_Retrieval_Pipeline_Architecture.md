# Multi-Source Retrieval Pipeline — Technical Architecture

**Role:** Lead Machine Learning Engineer  
**Platform:** Azure Databricks · Unity Catalog (shared access mode)  
**Stack:** PySpark ML (ALS), NumPy, pandas, `concurrent.futures`

---

## 1. Problem and goal

The existing single-model retrieval stage misses niche user interests and performs poorly for cold-start users. This pipeline replaces it with three complementary candidate generators — ALS collaborative filtering, a Two-Tower neural model and LightGCN graph propagation — merges their outputs into one globally deduplicated candidate pool, and measures with Recall@100 whether the blend retrieves more relevant items than any single source.

## 2. Pipeline at a glance

```mermaid
flowchart LR
    V[(Unity Catalog Volume<br/>lab_assets_vol)] --> L[Load + validate<br/>artifacts]
    L --> A[ALS candidates<br/>main thread]
    L --> P{{ThreadPoolExecutor<br/>max_workers=2}}
    P --> T[Two-Tower candidates]
    P --> G[LightGCN candidates]
    A --> N[Normalize item IDs<br/>canonical strings]
    T --> N
    G --> N
    N --> U[Global set union<br/>ALS ∪ Two-Tower ∪ LightGCN]
    U --> E[Recall@100 per source<br/>vs combined pool]
```

1. **Load and validate** five artifacts from a Unity Catalog Volume; fail fast on shape or schema problems.
2. **Generate** top-100 candidates per user: ALS synchronously on the main thread, then Two-Tower and LightGCN concurrently.
3. **Normalize** every item ID to one canonical string form.
4. **Deduplicate** with a single set union across all sources and users.
5. **Evaluate** Recall@100 per source against the combined pool, plus like-for-like diagnostics.

## 3. Artifacts and validation

| Artifact | Format | Loaded as | Validation before use |
|---|---|---|---|
| ALS model | Spark ML model directory | `ALSModel.load()` | Loads without deserialization error |
| Two-Tower item embeddings | `.npy` | NumPy array (n_items × d) | 2-D; row count equals ID count |
| Two-Tower item IDs | `.npy` | NumPy string array | Aligned 1:1 with embedding rows |
| LightGCN candidates | Parquet | pandas DataFrame | Has `user_id`, `item_id` |
| Ground truth interactions | Parquet | pandas DataFrame | Has `user_id`, `item_id`; non-empty |

**Why a Volume.** The ALS model is a Spark ML directory that Spark reads with distributed tasks, which needs a governed path that both the driver and executors can reach; the legacy DBFS root and workspace files don't support that on Unity Catalog clusters. The same `/Volumes/...` path also works with ordinary Python file APIs (`np.load`, `pd.read_parquet`), so one location serves both the Spark and non-Spark artifacts.

**Why validate embedding dimensions.** Row *i* of the embedding matrix must belong to item ID *i*. A count mismatch either raises an `IndexError` when IDs are looked up or, worse, silently maps scores to the wrong items and corrupts every recall number. A width mismatch breaks the user–item dot product with a shape error. Asserting both at load time turns these into one clear failure before any work is done.

**ID conventions.** `user_id` is an integer throughout; `item_id` is a string of the form `item_N`.

## 4. Candidate sources

| Source | Signal | What it adds to the blend | How candidates are produced |
|---|---|---|---|
| **ALS** | Matrix-factorization latent factors learned from interactions | Strong collaborative signal for warm users; cannot score users absent from training (cold start) | Dot product of the user's factor vector with every item factor, top-100 via `argpartition` |
| **Two-Tower** | Dense user and item embeddings from neural towers | Towers can embed new users or items from their features, which is the standard answer to cold start | Dot product with all item embeddings, top-100 via `argsort` |
| **LightGCN** | Embeddings propagated over the user–item graph | Multi-hop neighbours surface long-tail and niche items that direct co-occurrence misses | Pre-computed candidates filtered to requested users, first 100 per user |

**ALS implementation detail.** `recommendForUserSubset()` relies on Spark higher-order functions that Unity Catalog shared-access clusters block (`UC_COMMAND_NOT_SUPPORTED`). The pipeline instead reads `userFactors` and `itemFactors` directly, brings them to the driver with `.toPandas()`, and scores in NumPy. Each item factor row is mapped back to its ALS item `id` rather than assuming row position equals item ID, because Spark does not guarantee row order. Users with no learned factors are skipped, which is exactly the cold-start gap the other two sources are there to fill.

## 5. Orchestration — why ALS runs on the main thread

**ALS is a Spark workload.** `ALSModel.load()` and the two `.toPandas()` calls go through the SparkSession and the JVM (via Py4J) and submit Spark jobs. Under Databricks' pinned-thread execution model, each Python thread is paired with its own JVM thread, and Spark's per-thread state — job group, scheduler pool and other local properties — belongs to the thread that owns the session. A fresh `ThreadPoolExecutor` worker doesn't inherit that context, so Spark calls from it are unsafe. ALS therefore runs synchronously on the main thread before the pool starts.

**Two-Tower and LightGCN are pure driver-side computation.** Two-Tower is a NumPy matrix multiply and sort; LightGCN is a pandas filter and `groupby().head()`. Neither touches Spark, and both only read arrays already in memory, so running them side by side can't corrupt shared state. NumPy's matrix multiply releases Python's GIL, so the two genuinely overlap and the parallel stage costs roughly the slower of the two rather than their sum.

**Executor design.** `max_workers=2` gives one worker per source; more workers would add nothing. Each source is submitted with `executor.submit()`, and `.result()` is called on each future, which blocks until it finishes and re-raises any worker exception on the main thread so failures are never swallowed.

## 6. Deduplication strategy

- **Canonical string IDs first.** Every ID passes through one normalizer: bytes are decoded (`np.load` can return `b'item_7'`), values are cast to `str` and stripped, and bare integers are mapped onto the `item_N` convention. As a result `7`, `"7"`, `"item_7"` and `b"item_7"` all become the same key. Without this step a set treats them as different items and the pool double-counts.
- **One global union.** All three sources, across all users, are merged in a single operation: `set(als_ids) | set(tt_ids) | set(lgcn_ids)`. Deduplicating inside each source first would leave every cross-source duplicate in place.
- **Cost.** Set construction and union are hash-based, so the whole step is roughly linear in the number of candidates, versus quadratic for list-scan deduplication.
- **Trade-off: rank is lost.** A set has no order, so the combined pool is evaluated in full (`k=None`) rather than sliced at `[:100]`, which would keep an arbitrary 100 items. The per-source lists keep their rank order and are scored at their first 100 entries. A production blender that must hand an ordered top-K to the ranker would use rank-aware fusion (for example reciprocal rank fusion or per-source quotas) before truncating.

## 7. Evaluation method

**Metric.** Recall@k = |candidates[:k] ∩ ground truth| / |ground truth|, returning 0 for an empty ground-truth set. `k=None` scores the full candidate list.

**Required comparison.** Per-source Recall@100 versus Combined Recall on the full deduplicated pool, both against the global set of ground-truth items.

**Why supplementary views were added.** Each per-source list is flattened user by user, so its first 100 entries are in practice one user's top-100 measured against every user's ground truth, while the combined pool spans every user and all three sources. That comparison is structurally lopsided, so two like-for-like views are reported alongside it:

- **Global coverage (full pool):** each source scored on all of its candidates, the same `k=None` basis as the combined pool.
- **Mean per-user Recall@100:** each user's own top-100 from a source against that user's own ground truth, averaged over users. The combined row uses each user's union of three top-100 lists (up to 300 items).

A last column counts relevant items that only one source retrieved — the direct evidence of what each model uniquely contributes.

## 8. Results

> Values below come from the executed notebook (`Multi_Source_Retrieval_Pipeline_Assignment.py`). The supplementary cell prints this table in exactly this format, ready to paste.

| Source | Recall@100 (required metric) | Global coverage, full pool | Mean per-user Recall@100 | GT items only this source found |
|---|---|---|---|---|
| ALS | x.xxxx | x.xxxx | x.xxxx | n |
| Two-Tower | x.xxxx | x.xxxx | x.xxxx | n |
| LightGCN | x.xxxx | x.xxxx | x.xxxx | n |
| **Combined pool** | x.xxxx (full pool) | x.xxxx | x.xxxx (per-user pool, avg n items) | n/a |

**Headline.** Combined Recall on the full deduplicated pool is **x.xxxx**, a lift of **+x.xxxx** over the best single source (**[source]**) on the required metric and **+x.xxxx** on the like-for-like full-pool comparison.

**Interpretation.**

- **The lift is non-negative by construction.** The combined pool is the union of every source's candidates, so it contains every hit any single source made and cannot score lower than any of them on recall. Whether the lift is *strictly* positive depends on the "only this source found" column: relevant items retrieved by exactly one model are the proof that the three sources are complementary rather than redundant.
- **The required lift overstates the gain.** It compares roughly one user's top-100 against all users and all sources. The like-for-like column is the fair measure of what blending adds at the catalogue level.
- **Global coverage saturates on small catalogues.** Once (users × 100) is large relative to the number of items, every source's full pool touches almost the whole catalogue and all global numbers crowd toward 1.0. In that regime the mean per-user column is the meaningful lift measure, because it asks whether each individual user's pool contains their own relevant items.
- **Lab caveats.** Two-Tower user embeddings are seeded random vectors in this lab, so that source behaves as a random-query baseline: what it contributes is coverage, not learned relevance. ALS skips cold-start users, and LightGCN's first-100 selection assumes its parquet is already ranked within each user.

## 9. Latency design goal (not measured here)

The brief calls for a latency-conscious design. As a design goal — not something this notebook measures or enforces — the retrieval stage should fit a tight online budget, on the order of **≤ 50 ms p99** for all three sources combined. The notebook prints stage timings for reference only; batch timings on a shared cluster say nothing reliable about online serving latency.

How a production version would meet that goal:

- **Keep Spark out of the request path.** A scheduled Spark job exports ALS factors and Two-Tower item embeddings to an approximate-nearest-neighbour index (for example Databricks Vector Search, FAISS or ScaNN) so online lookups are sub-millisecond vector queries, not Spark jobs.
- **Pre-compute LightGCN** candidates offline into a low-latency key-value store keyed by user.
- **Fan out concurrently** to all three sources with a per-source timeout, and return whatever arrives within budget (graceful degradation), so stage latency tracks the slowest healthy source rather than the sum.
- **Cache hot users** and their merged pools.

If ALS ever had to run concurrently inside Spark itself, `pyspark.InheritableThread` (which carries the parent thread's Spark local properties into the child) is the supported route — but removing Spark from the serving path is the better fix.

## 10. Limitations and next steps

- **Rank-aware blending.** Replace the unordered union with reciprocal rank fusion or source quotas so the stage can emit a true ordered top-K.
- **Train the user tower.** Swap the random user embeddings for learned ones so Two-Tower contributes relevance, then re-measure its unique contribution.
- **Segment-level evaluation.** Report recall separately for cold-start, long-tail and head users — the segments the redesign targets — and add catalogue coverage and long-tail share alongside recall.
- **Scale ALS scoring.** Batch the per-user dot products (or move to the ANN index) once the user and item counts grow beyond what a per-user driver loop handles comfortably.
