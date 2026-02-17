Pipeline Concepts: build, reduce, and score
============================================

Bergson's CLI centers on three commands — ``build``, ``reduce``, and ``score`` — that
together implement gradient-based data attribution. This page explains what each command
does, what it produces, and when you should use each one.

Overview
--------

All three commands run the same underlying gradient collection pipeline:

.. code-block:: text

   raw gradient → apply normalizer → apply random projection → write or aggregate

The difference between them is **what they do with the collected gradients**:

- ``build`` writes a **per-example gradient** to an on-disk index.
- ``reduce`` **aggregates** all gradients from a dataset into a single vector and writes it to an on-disk file.
- ``score`` **computes similarity scores** by comparing gradients from one dataset against a pre-built query.

.. _build-command:

``build`` — Build a Per-Example Gradient Index
-----------------------------------------------

``build`` runs every example in your dataset through the model, collects a gradient for
each one, and stores the resulting vectors in a memory-mapped index on disk.

The index is keyed by example and supports fast nearest-neighbour search via
``bergson query``.

**Typical use cases**

- You want to find which training examples are most similar to a given query (e.g. an
  eval example or a generated output).
- You intend to query the index multiple times against different queries, so it's worth
  paying the up-front cost to store all gradients.
- You are using random projections (``--projection_dim > 0``) so each gradient is
  small enough to store individually.

**What it produces**

A directory at ``run_path`` containing:

- ``data.npy`` — a memory-mapped array of shape ``[num_examples, projection_dim]``
  (or ``[num_examples, param_dim]`` if no projection).
- ``indices.jsonl`` — per-example metadata.
- ``info.json`` — gradient shapes and dtypes.
- ``processor.pt`` (if normalizer or preconditioner is enabled) — the fitted
  ``GradientProcessor``.

**Example**

.. code-block:: bash

   bergson build runs/my-index \
       --model EleutherAI/pythia-160m \
       --dataset NeelNanda/pile-10k \
       --projection_dim 16

After building, use ``bergson query`` to interactively search the index:

.. code-block:: bash

   bergson query --index runs/my-index

.. note::

   Random projections (``--projection_dim > 0``) dramatically reduce per-example
   storage. With no projection (``--projection_dim 0``), storing per-example gradients
   is only practical for small models or small datasets.

.. _reduce-command:

``reduce`` — Aggregate a Dataset into a Single Query Gradient
-------------------------------------------------------------

``reduce`` collects per-example gradients and immediately **aggregates** them into a
single representative vector (mean or sum). Only the aggregate is written to disk, not
the individual per-example gradients.

The resulting aggregate is typically used as the **query** for a subsequent ``score``
run.

**Typical use cases**

- You are using no random projection (``--projection_dim 0``) and individual gradients
  would be too large to store.
- You want to compute the average influence of a dataset on another dataset (e.g.
  finding which training examples are relevant to an entire eval set).
- You want a compact query representation before running ``score``.

**What it produces**

A directory at ``run_path`` containing:

- ``data.npy`` — a single aggregated gradient vector of shape ``[1, param_dim]``.
- ``info.json`` — gradient shapes and dtypes.
- ``processor.pt`` (if normalizer or preconditioner is enabled) — the fitted
  ``GradientProcessor``.

**Key options**

- ``--method mean`` (default) or ``--method sum``: how to aggregate gradients.
- ``--unit_normalize``: unit-normalize individual gradients *before* aggregating.

**Example**

.. code-block:: bash

   bergson reduce runs/my-query \
       --model EleutherAI/pythia-160m \
       --dataset NeelNanda/pile-10k \
       --method mean \
       --unit_normalize \
       --projection_dim 0

.. note::

   ``--unit_normalize`` in ``reduce`` applies normalization *per example before*
   aggregating, so each example contributes equally to the mean direction regardless of
   gradient magnitude. This is different from normalizing the final aggregated vector
   (which would have no effect on downstream ranking). When using preconditioners,
   normalization must happen after preconditioning, which is done in ``score`` not
   ``reduce``.

.. _score-command:

``score`` — Score a Dataset Against Pre-Computed Query Gradients
----------------------------------------------------------------

``score`` computes a scalar influence score for every example in a dataset by comparing
its gradient against a set of pre-computed **query gradients** loaded from disk.

The query gradients were previously produced by ``reduce`` (or ``build``). The scoring
process in ``score`` applies preconditioning and normalization to the loaded query
gradients before computing dot products.

**Typical use cases**

- You have a query index (from ``reduce`` or ``build``) and want to rank a large
  training dataset by influence.
- You want to apply preconditioners (e.g. KFAC, EK-FAC, Adam second moments) to
  scale the gradient comparison.
- You don't need to store individual training gradients on disk — ``score`` computes
  and immediately discards each training gradient after comparing it.

**What it produces**

A directory at ``run_path`` containing:

- ``scores.npy`` — a memory-mapped array of shape ``[num_examples]`` (or
  ``[num_examples, num_queries]`` for ``--score individual``).
- ``info.json`` — scoring metadata.

**Scoring modes** (``--score``)

- ``mean`` (default): compare each training gradient to the *mean* of the query
  gradients. Useful when queries represent an aggregate evaluation signal.
- ``nearest``: compare each training gradient to the *most similar* query gradient
  (max over all queries). Useful when queries represent distinct individual examples.
- ``individual``: compute a separate score for every query gradient. Produces an
  ``[N_train, N_query]`` score matrix.

**Key options**

- ``--query_path``: path to the pre-computed query gradient index (required).
- ``--unit_normalize``: unit-normalize training gradients before scoring.
- ``--query_preconditioner_path``, ``--index_preconditioner_path``,
  ``--mixing_coefficient``: apply and optionally mix preconditioners to scale the
  gradient comparison.
- ``--modules``: restrict scoring to a subset of model modules.

**Example**

.. code-block:: bash

   bergson score runs/my-scores \
       --model EleutherAI/pythia-160m \
       --dataset EleutherAI/pile \
       --query_path runs/my-query \
       --score mean \
       --unit_normalize \
       --projection_dim 0

Choosing the Right Command
--------------------------

The decision tree below covers the most common scenarios:

.. code-block:: text

   Do you want to search a gradient index interactively (e.g. per-prompt)?
   ├── Yes → use build + query
   └── No  → Do you want to use full gradients without random projection or preconditioning?
             ├── Yes, and the query is a single dataset → use reduce (for query) + score
             └── Yes, and you may reuse the same index for many queries → use build + score

**Using random projections**

When ``--projection_dim > 0`` (the default), individual gradients are small and
``build`` is generally preferred because the resulting index can be reused for multiple
queries. When ``--projection_dim 0``, individual gradient storage is expensive, so
``reduce`` is a better fit for the query side.

**Using preconditioners**

When using preconditioners (KFAC, EK-FAC, Adam second moments), preconditioning is
applied in ``score`` to the loaded query gradients. The recommended pipeline is:

.. code-block:: text

   bergson hessian   → compute preconditioners (stored separately)
   bergson reduce    → aggregate query gradients (per example, no preconditioning)
   bergson score     → load query, apply preconditioners, score training data

Applying ``--unit_normalize`` in ``score`` (not ``reduce``) ensures normalization
happens *after* preconditioning.

Worked Example: LESS-style Query Influence
------------------------------------------

This example computes the influence of a training set on a small evaluation set,
following the style of `LESS <https://arxiv.org/pdf/2402.04333>`_.

**Step 1 — Build a preconditioner on training data**

.. code-block:: bash

   bergson hessian runs/preconditioner \
       --model EleutherAI/pythia-160m \
       --dataset my-train-data

**Step 2 — Reduce the eval set to a query gradient**

.. code-block:: bash

   bergson reduce runs/eval-query \
       --model EleutherAI/pythia-160m \
       --dataset my-eval-data \
       --method mean \
       --projection_dim 0

**Step 3 — Score training examples against the query**

.. code-block:: bash

   bergson score runs/scores \
       --model EleutherAI/pythia-160m \
       --dataset my-train-data \
       --query_path runs/eval-query \
       --query_preconditioner_path runs/preconditioner \
       --score mean \
       --unit_normalize \
       --projection_dim 0

The resulting ``runs/scores/scores.bin`` contains one score per training example.
Higher scores indicate stronger positive influence on the eval set.
