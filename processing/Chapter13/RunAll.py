import argparse
from pathlib import Path
import subprocess
import sys


def main(noshow: bool = False) -> int:
    """Run all Figure*.py scripts in this chapter."""
    chapter_dir = Path(__file__).resolve().parent
    scripts = sorted(
        p for p in chapter_dir.glob("Figure*.py") if p.name != Path(__file__).name
    )

    if not scripts:
        print(f"No Figure scripts found in {chapter_dir.name}.")
        return 0

    failed = 0
    for script in scripts:
        cmd = [sys.executable, script.name]
        if noshow:
            cmd.append("--noshow")
        print(f"[RUN] {chapter_dir.name}/{script.name}")
        result = subprocess.run(cmd, cwd=chapter_dir)
        if result.returncode != 0:
            failed += 1
            print(f"[FAIL] {script.name} (exit {result.returncode})")

    if failed:
        print(f"Completed with {failed} failure(s).")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Disable interactive figure display for all scripts.",
    )
    args = parser.parse_args()
    raise SystemExit(main(noshow=args.noshow))
