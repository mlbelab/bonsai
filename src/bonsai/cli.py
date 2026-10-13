"""Command-line interface: `bonsai run` (upstream bonsai_main.py) and `bonsai config`."""

import argparse
import logging

FORMAT = "%(asctime)s %(funcName)s %(levelname)s %(message)s"  # copy upstream FORMAT exactly


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `bonsai` command."""
    parser = argparse.ArgumentParser(prog="bonsai")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Reconstruct a Bonsai tree from a YAML config.")
    from ._core.helpers import str2bool

    run_p.add_argument(
        "--config_filepath",
        type=str,
        default=None,
        help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
        "needed to run Bonsai.",
    )
    run_p.add_argument(
        "--step",
        type=str,
        default="all",
        help="Optional: Use this argument to run all or only a single step of the script, continuing from "
        "the state of the previous step. Can be used when parallelizing on HPC-clusters, because "
        "mostly core_calc benefits from parallelization, and preprocess has the highest memory "
        "requirements."
        "Allowed values (in order of execution): all, preprocess, core_calc, metadata.",
    )
    run_p.add_argument(
        "--pickup_intermediate",
        type=str2bool,
        default=False,
        help="Optional: Set this argument to true if Bonsai needs to search in the indicated results-folder"
        "for the furthest developed tree reconstruction, and pick it up from there. If this argument"
        "is True, it over-rules the same argument in bonsai_config.yaml, and the other way around.",
    )
    run_p.add_argument(
        "--store_all_nwk_folder",
        type=str,
        default="",
        help="REMOVE LATER! This will slow down the program by storing a newick file after every change to"
        "the tree. This is only necessary for making a tree reconstruction animation.",
    )
    run_p.add_argument(
        "--print_annotations",
        type=str,
        default="",
        help="REMOVE LATER! Only for debugging, string given here should give path to annotation-folder.",
    )
    sub.add_parser(
        "config", add_help=False, help="Create or inspect a Bonsai YAML config file (forwards to upstream's CLI)."
    )  # forwards remaining args to _core.config.main
    args, rest = parser.parse_known_args(argv)
    logging.basicConfig(format=FORMAT, datefmt="%m-%d %H:%M:%S", level=logging.WARNING)
    # upstream's own "myapp" logger ended up at DEBUG (its WARNING default is immediately
    # overwritten, see docs/deviations.md); match that effective level exactly, not INFO.
    logging.getLogger("myapp").setLevel(logging.DEBUG)
    if args.command == "config":
        from ._core.config import main as config_main

        config_main(rest)
        return 0
    if rest:
        parser.error("unrecognized arguments: {}".format(" ".join(rest)))
    from ._core.helpers import Run_Configs
    from ._core.main import run

    cfg = Run_Configs(args.config_filepath, step=args.step)
    if args.pickup_intermediate:
        cfg.pickup_intermediate = True  # type: ignore[attr-defined]  # set dynamically in Run_Configs.__init__
    run(cfg, args.store_all_nwk_folder, args.print_annotations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
