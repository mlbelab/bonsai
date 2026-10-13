# Deviations from upstream (b16ef6f)

## Known upstream bugs kept unchanged (phase 1)
1. `_core/tree_helpers.py` (TH:1705, TH:1761): `ind != inds_to_be_del` compares an int
   with a list, so the filter never removes anything; pairs with UBs are recomputed (slower, same result).
2. `_core/tree_helpers.py` TH:1237: `indTask + 1 % 100` parses as `indTask + 1`, so the periodic MPI max exchange
   never fires.
3. `_core/tree_helpers.py` TH:378, TH:4918; `_core/clustering.py` (raise lines CL:698, CL:857);
   `_core/config.py` CG:161; `_core/helpers.py` (7 sites: HL:105, HL:1300-1352);
   `_core/dataprocessing.py` (17 sites); and 3 normal-termination sites in `_core/main.py` (now
   `return`): library-terminating `exit()`/`sys.exit()` calls; the error-path ones are replaced by
   `raise BonsaiError(...)` (error path only, same message where one preceded the call). Note: the
   brief's own locator command (`grep -n "exit()"`) only finds calls with empty parentheses and
   counts 27; the actual upstream file also has 8 live `exit("message")`/`sys.exit("message")` call
   sites (dataprocessing.py 7, tree_helpers.py 1) that are semantically identical
   abnormal-termination calls. All of them (35 total, 3 of which are the normal-termination ones in
   `main.run`) were converted; none of this touches numerics, decision rules, iteration order or
   RNG.
   - **Fix round 1:** two `except Exception` handlers whose `try` body can reach a converted
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
4. SPR random choices (TH:2403, 4403, 4451) use the unseeded global NumPy RNG (the NNI seed
   `np.random.seed(42)` is set later). The golden fixture was reproducible across runs, but larger
   inputs may not be deterministic.

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
  explicitly (**fix round 1**: was mistakenly `logging.INFO` and the task-2 report wrongly claimed importing
  `_core.helpers` afterwards would reset it to DEBUG anyway and self-correct the visible level; measurement
  showed the effective level stayed at INFO (20), hiding `mp_print(..., DEBUG=True)`/`logger.debug` output
  that upstream always showed).
- Matplotlib is now imported lazily in `_core/helpers.py` (`_plotting_setup()`, `_plot_colors()`), used only by
  plotting functions (`get_celltype_colors`, `pause_anim`, `_core/dataprocessing.py`'s `set_node_edge_style`);
  `import bonsai` no longer loads matplotlib.
- `tl.node_posteriors` reads the posterior `.npy` files that upstream already wrote to the kept
  `reconstruct(..., keep_workdir=True)` work directory, instead of recomputing the posterior from the
  stored tree. Recomputation is cheap and is left to a later (phase 2) milestone; for now the function
  raises `ValueError` if the AnnData does not carry a kept workdir (`uns["bonsai"]["workdir"]` falsy).
