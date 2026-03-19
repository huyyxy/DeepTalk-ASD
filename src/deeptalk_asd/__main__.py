"""
DeepTalk-ASD CLI 工具

Usage:
    python3 -m deeptalk_asd download-models [--cache-dir DIR]
    python3 -m deeptalk_asd info [--cache-dir DIR]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="deeptalk_asd",
        description="DeepTalk-ASD: Active Speaker Detection CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download-models
    dl_parser = subparsers.add_parser(
        "download-models",
        help="Download all required model files to the cache directory",
    )
    dl_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: ~/.cache/deeptalk_asd/)",
    )

    # info
    info_parser = subparsers.add_parser(
        "info",
        help="Show model cache status and configuration",
    )
    info_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory to inspect",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    from .model_manager import ensure_all_models, print_model_info

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.command == "download-models":
        ensure_all_models(cache_dir)
    elif args.command == "info":
        print_model_info(cache_dir)


if __name__ == "__main__":
    main()
