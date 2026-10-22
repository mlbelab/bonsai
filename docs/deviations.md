# Deviations from upstream (b16ef6f)

## Known upstream bugs kept unchanged (phase 1)
1. `_core/tree_helpers.py` (TH:1705, TH:1761): `ind != inds_to_be_del` compares an int
   with a list, so the filter never removes anything; pairs with UBs are recomputed (slower, same result).
2. `_core/tree_helpers.py` TH:1237: `indTask + 1 % 100` parses as `indTask + 1`, so the periodic MPI max exchange
   never fires.
3. `_core/tree_helpers.py` TH:378, TH:4918; `_core/clustering.py` (raise lines CL:698, CL:857);
   `_core/config.py` CG:161; `_core/helpers.py` (7 sites: HL:105, HL:1300-1352); and
   `_core/dataprocessing.py` (21 sites): library-terminating `exit()`/`sys.exit()` calls
   (including `exit("message")`/`sys.exit("message")` call sites, which are semantically
   identical abnormal-termination calls) replaced by `raise BonsaiError(...)` (error path only,
   same message where one preceded the call) — 33 sites total
   (`grep -rc "raise BonsaiError" src/bonsai/_core/*.py`: dataprocessing.py 21, helpers.py 7,
   tree_helpers.py 2, clustering.py 2, config.py 1). Three further normal-termination `exit()`
   sites in `_core/main.py` (`main.py:114`, `main.py:417`, `main.py:539`) are now plain `return`.
   None of this touches numerics, decision rules, iteration order or RNG. Packaging consequence:
   the CLI (`bonsai run`) does not catch `BonsaiError`, so an error now exits with code 1 and a
   Python traceback, where upstream's bare `exit()` (a `SystemExit()` with no argument) exited 0.
   - Two `except Exception` handlers whose `try` body can reach a converted
     `raise BonsaiError(...)` used to let upstream's `exit()` (`SystemExit`, a `BaseException`)
     propagate unimpeded, but would now silently swallow the `BonsaiError` (a `ValueError`,
     `Exception` subclass) instead, changing abort-vs-continue behaviour. Fixed by adding
     `except BonsaiError: raise` immediately before each: `dataprocessing.py:2638` (guards the "no
     data was found" raise reached from `load_data_for_tree`'s `else` branch, now covered by
     `tests/test_load_data_for_tree.py`) and `main.py:80` (guards the `--pickup_intermediate` retry
     path, which could otherwise swallow e.g. the nCells/cellIds mismatch `BonsaiError` raised from
     `Metadata.__init__` and silently fall back to re-preprocessing). The third broad handler,
     `dataprocessing.py:1025`, was checked and its `try` body (a `pandas` `reindex`/shape check,
     annotation-file salvage logic) cannot reach any converted `raise`, so it was left unchanged.
     Narrower handlers (`except ValueError`, `except FileNotFoundError`, `except KeyError`, etc.)
     were also checked; none of their `try` bodies call into converted code. Bare `except:` clauses
     already caught `SystemExit` under upstream too, so they are unchanged.
4. In the default pipeline, every random draw that affects the reconstructed tree or likelihood is
   explicitly seeded. `Tree.getOptimalDLogLPairInfoNew`'s `np.random.choice(posDLogLInds, ...)`
   (`tree_helpers.py:2403`) only executes when its `random` argument is `True`; that argument is
   `True` only on the call path from `_core/dataprocessing.py:nnnReorderRandom` (defined at
   `dataprocessing.py:1771`), which calls `Tree.getOptEdgeList(..., random=True, ...)`
   (`dataprocessing.py:1878`). `getOptEdgeList` forwards `random=True` directly into
   `Tree.mergeChildrenUB(..., random=random)` (`tree_helpers.py:3725`), which forwards it in turn
   into `getOptimalDLogLPairInfoNew(..., random=random, ...)` (`tree_helpers.py:1409`). Every call
   into `nnnReorderRandom` happens after `np.random.seed(42)` (`main.py:390`, and again at
   `dataprocessing.py:1829` inside `nnnReorderRandom` itself) and a further per-task reseed
   `np.random.seed(seeds[task])` (`dataprocessing.py:1848`), where `seeds` is itself computed
   deterministically from that same `seed(42)` state (`dataprocessing.py:1830`,
   `np.random.choice(int(randomTries * 1e4), size=randomTries, replace=False)`). `Tree.do_spr_moves`'s
   `select_cand`/`select_target='random'` branches (`tree_helpers.py:4403`, `tree_helpers.py:4451`)
   also draw from the global RNG unseeded by this mechanism, but the default pipeline never reaches
   them: `spr_strategy="large_tree"`'s branch of `do_spr_moveset` (the `tl.reconstruct`/CLI default)
   calls `do_spr_moves_with_postprocessing(..., select_cand='long_branches_first',
   select_target='cluster_centers', ...)` (`dataprocessing.py:1384-1389`), neither selector being
   `'random'`.

   The only remaining unseeded global-NumPy draws are cosmetic: random suffixes used to build
   temporary folder/file names that do not feed into any numeric result or decision
   (`helpers.py:1382`, `helpers.py:1384`, `tree_helpers.py:2011`, `tree_helpers.py:2325`,
   `tree_helpers.py:2327`). `globals.reset()` deliberately does not reseed between runs, so a
   second call in the same process could pick different folder-name suffixes than the first, but
   this does not change the tree, branch lengths or likelihood.

## Validation deviations (phase 1 packaging)
5. `bonsai._io_folder.validate_inputs` is stricter than upstream in two ways it does not relax:
   non-finite mean/SD/prior-variance/mu is always refused (upstream would propagate NaN/Inf
   silently into the likelihood), and a negative SD is refused (upstream's own derivation assumes
   `sd >= 0`). It is not stricter anywhere else: `sd == 0` and non-positive
   `var["bonsai_prior_var"]` are accepted and passed through unchanged, matching upstream's own
   handling (it clamps `var = max(sd**2, 1e-12)` and drops genes where
   `prior_var - var < 1e-4`, rather than rejecting them up front).
6. `bonsai.tl._check_names` checks gene names only for uniqueness and the absence of a tab or
   newline (which would corrupt the per-gene TSV columns they index); gene names are never used as
   newick labels, so they do not need the full cell-name check. Cell names keep the full check:
   uniqueness, no whitespace or `(),:;` (newick syntax characters), and no collision with upstream's
   internal vertex names (`root`, `internal_<n>`).

## Packaging changes
- Config paths are relative to the current directory (upstream chdir'd to the repo root); this also applies to
  `_core/config.py`'s YAML-template lookup, which previously relied on the chdir and now resolves
  `config_template.yaml` next to `config.py` via `Path(__file__).with_name(...)`.
- Logging is configured by the CLI only (`bonsai.cli.main`); library users call `logging.basicConfig` themselves.
  The module-level `logging.basicConfig(...)` calls in `_core/helpers.py` and `_core/dataprocessing.py` were
  removed; `logger = logging.getLogger("myapp")` and its `setLevel(...)` call are unchanged. Upstream's own
  `log_level = logging.WARNING` is immediately overwritten by `log_level = logging.DEBUG` in both files, so the
  "myapp" logger's effective level was always DEBUG (10), not WARNING, regardless of the root `basicConfig`
  level. `bonsai.cli.main` reproduces this by setting `logging.getLogger("myapp").setLevel(logging.DEBUG)`
  explicitly.
- Matplotlib is now imported lazily in `_core/helpers.py` (`_plotting_setup()`, `_plot_colors()`), used only by
  plotting functions (`get_celltype_colors`, `pause_anim`, `_core/dataprocessing.py`'s `set_node_edge_style`);
  `import bonsai` no longer loads matplotlib.
- `tl.node_posteriors` reads the posterior `.npy` files that upstream already wrote to the kept
  `reconstruct(..., keep_workdir=True)` work directory, instead of recomputing the posterior from the
  stored tree. Recomputation is cheap and is left to a later (phase 2) milestone; for now the function
  raises `ValueError` if the AnnData does not carry a kept workdir (`uns["bonsai"]["workdir"]` falsy).
- `tl.reconstruct`'s `verbose` defaults to `False`; the upstream config template defaults it to `True`.
  `verbose` only gates extra `mp_print`/logger output inside `_core` (e.g. `main.py:148`, `main.py:157`,
  `storeTreeInFolder(..., verbose=args.verbose)` calls); it does not change any numeric result, decision
  rule or iteration order.

## Phase 2 fp64 references (`bonsai._ref.pair`, not used by `_core`)
- Tight optimiser: the reference 3-leaf solves define the optimum: the exact optimum of upstream's objective
  (to fp64 resolution), with upstream's own solves (tol 1e-4/1e-9) within their stopping gap of it. They run
  L-BFGS-B at machine precision (ftol 1e-15, gtol 1e-11), then a projected Newton polish (forward-FD Hessian of the analytic gradient),
  restarting until every free coordinate has |g_i| ≤ (16 + G)·eps64·Σ_g|terms of g_i|, i.e. a gradient
  that is zero to its own fp64 rounding. L-BFGS-B alone stops where f's decreases fall below f's rounding,
  which can leave |g| ~ 1e-6, and it crawls on the flat tail of a log t heading for LOG_T_MIN (gradient ∝ t);
  the polish takes such a coordinate onto its bound. Upstream scores pairs at tol 1e-4 (TH:1303). The
  2-leaf time uses brentq at rtol 4·eps64 instead of tol 1e-4/1e-9.
- Fallback trigger: upstream's sequential mode falls back to the joint solve when scipy reports
  `success=False` at its loose tolerance (TH:4867, 5381-5385). The reference falls back when its
  resolution-scaled KKT test still fails after 5 restarts (scipy's flag is not used), or when the t_12
  bracket fails. On the 1320 sequential pairs of the kernel test families (rng 100-119), neither falls back
  (upstream at tol 1e-4 and 1e-9). The earlier absolute test, 1e-8·(1 + |f|), fell back on 5 of them where
  upstream did not.
- 2-leaf bracket: upstream raises `BonsaiError` once f(ub) > 1e9 during the ×10 bracket expansion
  (TH:4917-4919). The reference has no such check: it returns (None, not converged) once lb > 1e6, and
  `pair_score` falls back to the joint solve. This is reachable only for Σ_g d² ≳ 1e11 (e.g. x1 = 0,
  x2 = 2e5, v = 0.1, G = 4), never for normalised data.
- `pair_score` raises `ValueError` when the parent has no third child, i.e. W_r − w_1 − w_2 ≤ 4·eps·W_r.
  Upstream would continue with W_R ≈ 0 and produce NaN or inf.

## Phase 2 `_engine` deviations (device kernels)
- Pair-solver convergence and fallback (A11): upstream's sequential 3-leaf solve falls back to the
  joint solve when scipy reports `success=False` at its own loose tolerance (`_core/tree_helpers.py`
  TH:4867, TH:5382-5386). The device solver instead runs a fixed number of damped-Newton iterations
  and decides convergence from a resolution-scaled projected-gradient (KKT) test on the result,
  never from a solver status flag. Damping is adaptive Levenberg–Marquardt (λ·|diag H|, growing on
  a rejected step, shrinking on an accepted one, capped by a Gershgorin bound), replacing a fixed
  Gershgorin shift that missed its own convergence test on 0.5–0.8% of pairs because the shift was
  15–50× too large in flat valleys. A step is accepted if the objective does not rise beyond its
  derived rounding bound, or if the relative projected gradient drops while the objective rises by
  at most a small, derived multiple of that bound (a strict-decrease rule stalls in float32; an
  r-only rule, tried first, let the solver accept an uphill jump to a worse KKT point). Any pair
  that still fails its convergence test after the fixed iteration budget is returned with an
  explicit not-converged flag, never silently as a fallback or an unconverged value treated as
  final; every flagged pair must be re-scored on the host with the fp64 reference before any merge
  decision uses it.
- Box handling in the device branch-length optimiser (A12): upstream's L-BFGS-B clips the iterate
  into its box and reports convergence once the clipped gradient's zero-slope rule fires. Clipping
  the iterate was tried for the device optimiser and rejected because it stalls L-BFGS; the device
  optimiser instead keeps the gradient component that still points back into the box at a clipped
  point (scipy's own projected-gradient, "pgtol", semantics) and only reports convergence there, or
  when the optimiser's own small-step/small-change (Cauchy) rule fires (next entry). A plain
  zero-slope rule at the box boundary was tried first and reported convergence at non-KKT points,
  where a further step would have moved the objective by a large, easily-measured amount.
- The float32 branch-length optimiser cannot reach upstream's default gradient tolerance (1e-6):
  the objective's own float32 resolution is reached first, and optimistix's small-step/small-change
  stopping rule (not the gradient-tolerance rule) fires at that point instead.
- A child whose posterior position exactly equals its parent's as-if-root mean has no direction to
  report (a 0/0 division, `_core/tree_helpers.py` TH:2136-2137): upstream's NumPy array crashes
  when the result (a NaN) reaches the nearest-neighbour library's input check. The device-kNN
  coordinate function returns the zero vector for that child instead (distance 1 to every unit
  vector, 0 to another zero vector), avoiding the crash without changing any other child's result
  (A13).
- The pair solver scores each 3-leaf subproblem to its solver's own convergence tolerance (near the
  true optimum, to float32/float64 resolution), where upstream scores it to a fixed tolerance
  (1e-4) during candidate search (TH:1303). Because the search is otherwise a free choice
  among positive-gain pairs, the engine's score for a given pair is typically at least upstream's
  score for the same pair (up to rounding, when both reach the same optimum), so it can pick a
  different, higher-scoring pair and so a different merge order than upstream on the same data.
- When neither the sequential nor the joint 3-leaf solve converges for a pair, the engine returns
  the sequential solve's iterate flagged not converged, so a later host re-score can still use it;
  upstream's pair score in that situation is `-inf` and the pair is treated as having no finite
  value.
- Block-scheduled tree pass (A14): the device tree pass no longer pads every tree level to the
  widest level, which scheduled 9–19× the useful rows on the trees measured (8,000-leaf random
  tree; two real trees of 772 and 1,194 nodes). The host now builds a schedule of blocks of static
  width B = min(256, bucketed n), ordered by level (post-order levels for the up pass, reversed for
  the down pass); each block holds at most B nodes of a single level, padded and masked, so a pass
  costs about (n + levels·B)·G rows. Sibling groups that span blocks are completed with carried and
  per-block tail partial sums, so every message is still formed by sums only, without subtraction.
  The compile key is (B, bucketed n, bucketed number of leaves, bucketed block count, G),
  independent of the tree's shape. The leaf data are stored once per leaf, (bucketed number of
  leaves, G), with a node-to-row index, instead of one row per node. The maths, the objective and
  every decision rule are unchanged: the existing agreement tests against the fp64 reference, the
  golden tree and the x64 checks pass at their unchanged, derived tolerances. Measured:
  scheduled/useful rows per pass 1.33/1.26 (up/down) on the 8,000-leaf tree, 1.04/1.03 on a
  100,000-leaf random tree, but 9–10 on the two real trees measured, and up to ~29 on deeper real
  trees of similar size, where the number of levels times B dominates. Memory, from CPU
  `memory_analysis` at 10,000 leaves × 2,000 genes, in units of one padded-nodes × genes float32
  array: tree pass 10.3 → 6.1, times-optimiser objective 9.3 → 6.0. Indicative GPU timing (one
  L40S, not a benchmark): tree pass 85 → 27 ms at 10,000 leaves and 1,952 → 217–228 ms at
  100,000 leaves, peak memory 20.9 → 14.8 GB at 100,000 leaves.

## Phase 2 M2 greedy engine (`bonsai._engine.greedy`, `bonsai._engine.build`, reference `bonsai._ref.greedy`)
The greedy tree builder searches freely for the same objective as upstream's greedy stage (`mergeChildrenUB`,
`_core/tree_helpers.py` TH:1039-1620, and redo starry, `mergeChildrenRecursive`, TH:968-1012). It does not replay
upstream's merge sequence step by step; the trees are compared by likelihood and topology after redo starry. The
fp64 reference `bonsai._ref.greedy` implements the same rules as the engine, so the entries below apply to both
unless stated. Reproduced from upstream, not deviations: the 1e-6 merge threshold (TH:2428), the periodic star
re-optimisation at the root every max(⌊0.05·C⌋, 10) merges (TH:1154, 2354), the add-as-child condition and order
(TH:2467-2476), the kNN threshold of 50 children and k = 10, and the bounds-off rule for ≤ 10 children (TH:1040).
- No zero-root branch. Upstream keeps a second running maximum for pairs whose new node lands on the parent
  (t_aR ≤ 1e-6, TH:1342-1347) and, when such a pair is best, re-optimises the star (root) or resets the child times
  (`resetChildTimes`, other nodes) instead of merging (TH:2381-2456, 2649-2686). It applies the test to the t_aR of its
  loose scoring solve (tol 1e-4, TH:1303, 4853-4867), which stays far above 1e-6: re-scoring 412 such pairs (30 seeds,
  8 cells, 30 genes) with upstream's own `calcSingleDLogL` gave t_aR > 1e-6 for all of them, so upstream merges them
  as ordinary pairs and the branch practically never fires. The engine's solver reaches the exact optimum (often
  t_aR ≈ 0), where the branch would fire often and change the search. The engine therefore drops it: a round merges
  the best pair iff its dLogL > 1e-6, whatever its t_aR. Emulating the loose solve was rejected because it cannot be
  reproduced exactly.
- Add-as-child as a zero edge, deferred collapse. When t_1a < 1e-6 and child 1 is internal, upstream makes child 2 a
  child of child 1 and then tries one extra merge of child 2 among child 1's children (`mergeDownstreamChildrenSingle`,
  TH:2467-2476, 2636-2641); likewise for child 2. The engine creates the new node anyway and sets that edge to exactly
  0, with the same condition and order (child 1, the lower id, first). A zero edge passes child 1's message unchanged,
  so the parent's state equals upstream's. Redo starry first collapses every zero edge above an internal non-cell node
  (upstream's `mergeZeroTimeChilds`, TH:789-807), which turns child 1's children plus child 2 into one polytomy that is
  resolved over all pairs; this replaces upstream's single extra merge, which is dropped. For t_1a = 0 this is the same
  tree; for 0 < t_1a < 1e-6 it moves that time by less than 1e-6, as upstream itself does.
- Repeated redo-starry passes. Upstream runs one redo-starry pass. A pass can leave new zero edges (a snap as above,
  an exact-zero optimum, or a star re-optimisation that puts an internal child's time at 0), so the engine repeats
  collapse + pass while any live internal non-cell node has t == 0, up to 8 passes; passes after the first run the
  greedy only at the nodes the collapse gave children. Every merge raises the likelihood by more than 1e-6, so the
  passes terminate; hitting the cap emits a `RuntimeWarning` and returns the (valid) tree with those zero edges. The
  greedy-stage tree is written from a copy with zero edges collapsed, as upstream's `cleanup_tree` does (DP:354-355).
- Redo-starry visiting order. Like upstream's recursion (children before their parent, TH:902-929, 968-1012), redo
  starry visits the nodes in post-order, but it takes the children by increasing node id where upstream takes them
  in its child-list order. That list is insertion order, and after `mergeZeroTimeChilds` the promoted grandchildren
  are appended at its end (TH:806). The order matters beyond ties: upstream updates the parent's as-if-root message
  after each child that changed (`getLtqsAfterChildUpdate`, TH:5353-5358), so the order decides which earlier
  siblings' changes a later sibling's outside message already contains, and with it that sibling's merges. The
  engine updates its messages in the same way: each node's outside message comes from its parent's float64 sums when
  the traversal reaches it, and a finished child's contribution is replaced in those sums. The reference recomputes
  every message from scratch instead; both agree in float64 and visit in the same order.
- Lazy-greedy bounds on the parent's actual movement. Upstream bounds a stored pair score by its slopes times a
  fixed ellipsoid size (`UB_ellipsoid_size`, TH:5118-5215). The engine applies the same slopes to how far the parent
  has actually moved since the pair was scored: every merge or star re-optimisation adds
  δx = ‖(x_new − x_old)·√W_old‖₂ and δW = ‖(W_new − W_old)/W_old‖₂ to running totals, and a pair scored at totals
  (Dx_e, DW_e) has the bound d + a·(Dx − Dx_e) + b·(DW − DW_e). The bound is still a heuristic, not a proof. Any change
  of a child's time, and every round when bounds are off (≤ 10 children), sets every bound to +inf, so those rounds
  score every candidate. A fresh score equal to the largest stale bound is accepted; upstream stops its scan only when
  the best score is strictly larger (TH:1257).
- kNN candidates. As upstream, candidate pairs come from the k = 10 nearest neighbours while the parent has more than
  50 children (TH:1888, 2115-2167) and from all pairs below. Differences: a new node's pairs come from an exact query
  over the current children, where upstream queries a stale index for 20·k neighbours; the index is rebuilt after
  max(⌊0.1·C⌋, 10) merges including the first time (upstream's first threshold has no floor of 10, TH:2355), after
  every star re-optimisation (TH:1417), and once more before stopping (upstream stops, TH:1435-1451); upstream's
  top-up of neighbours for sparsely paired nodes is not reproduced.
- Several merges per round (opt-in). With `merges_per_round` m > 1 (default 1; root greedy only, never in redo
  starry), a round merges the certified best pair plus up to m − 1 further pairs. These are freshly scored at the
  same parent state, have dLogL > 1e-6, are disjoint from each other and from the best pair, and are taken by
  decreasing dLogL with their stored times. Upstream merges one pair per round. The later pairs of a round are not
  re-scored after the earlier merges, so this is a free-search approximation whose default is decided by the M2
  gates.
- Periodic star re-optimisation on any round. Upstream checks the 5% schedule only on rounds that did not use bounds
  (TH:1926-1927); the engine checks it on every round.
- Merge times. Upstream re-solves the chosen pair at tol 1e-9 before merging because it scores at tol 1e-4. The
  engine merges with the times of the score itself: the device solve, converged to its own resolution, or the float64
  host re-score for flagged pairs.
- Host float64 weights. The parent's sums are float64 on the host. Each
  child weight is recomputed there as 1/(v + t) from the stored compute-dtype message and the float64 time, so the
  weight removed at a merge is bitwise the one added, without copying weights back from the device.
- Near-threshold accept decisions in float32 are made in float64, one-sided. In float32 a pair's dLogL carries a
  rounding error of order eps32·Σ|terms|, which on noisy data (cell variances ~1e6) exceeds the 1e-6 threshold: the
  engine would merge pairs that float64 and upstream reject. Before a merge is accepted (including the extra pairs of
  a `merges_per_round` round), a pair whose device score d satisfies d − 1e-6 ≤ its derived float32 resolution
  (1.25 × the rounding bound of its two star log-likelihoods, the ¼ covering the inputs' cast to float32) is
  re-scored on the host in float64, and the decision uses that score; if it is still the best over every bound, it is
  merged. Only false accepts are guarded: a pair whose device d ≤ 1e-6 while its float64 d > 1e-6 is not re-scored, so
  the engine can stop where float64 would merge once more. The M2 gate runs count such cases.
- Star EM fallback start. When the star EM's fixed-point sweeps do not converge, upstream falls back to BFGS on log t
  starting from log t (TH:5009), which is −inf for a child whose time was clamped to 0. The reference starts such
  children at log 1e-4 (upstream's own floor elsewhere, TH:4823) instead; the device star EM hands every
  non-converged star to that reference.
