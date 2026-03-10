import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing._runner import parse_args, run_dip_method


def main(show: bool = True) -> None:
    run_dip_method("figure53", "Figure53", show=show, chapter_attr="chapter05")


if __name__ == "__main__":
    args = parse_args()
    main(show=not args.noshow)
