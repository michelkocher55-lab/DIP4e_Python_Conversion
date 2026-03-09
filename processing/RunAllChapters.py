import argparse
from pathlib import Path
import subprocess
import sys


def main(noshow: bool = False) -> int:
    """Run all chapter RunAll.py scripts."""
    processing_dir = Path(__file__).resolve().parent
    chapter_dirs = sorted(p for p in processing_dir.glob("Chapter*") if p.is_dir())

    if not chapter_dirs:
        print("No chapter directories found.")
        return 0

    failed = 0
    for chapter_dir in chapter_dirs:
        runner = chapter_dir / "RunAll.py"
        if not runner.exists():
            print(f"[SKIP] {chapter_dir.name} (missing RunAll.py)")
            continue

        cmd = [sys.executable, runner.name]
        if noshow:
            cmd.append("--noshow")
        print(f"\n[CHAPTER] {chapter_dir.name}")
        result = subprocess.run(cmd, cwd=chapter_dir)
        if result.returncode != 0:
            failed += 1
            print(f"[CHAPTER FAIL] {chapter_dir.name} (exit {result.returncode})")

    if failed:
        print(f"\nCompleted with {failed} failing chapter(s).")
        return 1

    print("\nAll chapters completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Disable interactive figure display for all chapter scripts.",
    )
    args = parser.parse_args()
    raise SystemExit(main(noshow=args.noshow))
