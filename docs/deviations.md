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
