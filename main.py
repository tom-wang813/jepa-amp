import argparse
import subprocess
import sys


def run_module(module: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *extra_args]
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility entrypoint for JEPA-AMP workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("pretrain", "finetune", "finetune-supervised"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--gpu", type=int, default=0)

    for name in ("eval", "rep-eval"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--gpu", type=int, default=0)

    subparsers.add_parser("download-data")
    subparsers.add_parser("prepare-data")

    args = parser.parse_args()

    if args.command == "download-data":
        code = subprocess.call([sys.executable, "scripts/download_data.py"])
    elif args.command == "prepare-data":
        code = subprocess.call([sys.executable, "scripts/prepare_amp_dataset.py"])
    elif args.command == "pretrain":
        code = run_module("src.train.pretrain", ["--config", args.config, "--gpu", str(args.gpu)])
    elif args.command == "finetune":
        code = run_module("src.train.finetune", ["--config", args.config, "--gpu", str(args.gpu)])
    elif args.command == "finetune-supervised":
        code = run_module("src.train.finetune_supervised", ["--config", args.config, "--gpu", str(args.gpu)])
    elif args.command == "eval":
        code = run_module("src.eval.run_eval", ["--gpu", str(args.gpu)])
    else:
        code = run_module("src.eval.rep_eval", ["--gpu", str(args.gpu)])

    raise SystemExit(code)


if __name__ == "__main__":
    main()
