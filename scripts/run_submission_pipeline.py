"""Single entrypoint to rebuild reproducible submission artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission_pipeline import SubmissionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run submission pipeline steps.")
    parser.add_argument("--validate-only", action="store_true", help="Only run bundle validation and exit.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip required-file validation.")
    parser.add_argument("--skip-projection", action="store_true", help="Skip projection rebuild step.")
    parser.add_argument("--skip-report-pack", action="store_true", help="Skip report pack generation step.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit analysis-v1 run directory used by report-pack step.",
    )
    args = parser.parse_args()

    pipeline = SubmissionPipeline(project_root=ROOT)

    if not args.skip_validation:
        print("[0/2] Validating reproducibility bundle...", flush=True)
        status = pipeline.validate_bundle()
        print(status.to_string(index=False))
        pipeline.assert_bundle()
        print("Validation passed.\n", flush=True)
    else:
        print("[0/2] Validation skipped.\n", flush=True)

    if args.validate_only:
        return

    if not args.skip_projection:
        print("[1/2] Rebuilding projection views...", flush=True)
        pipeline.run_projection_rebuild()
    else:
        print("[1/2] Projection rebuild skipped.", flush=True)

    if not args.skip_report_pack:
        print("[2/2] Preparing report output pack...", flush=True)
        pipeline.run_report_pack(run_dir=args.run_dir)
    else:
        print("[2/2] Report pack step skipped.", flush=True)

    print("\nPipeline output status:", flush=True)
    print(pipeline.describe_outputs().to_string(index=False))


if __name__ == "__main__":
    main()
