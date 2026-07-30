#!/usr/bin/env python3
"""Build the report pages: the hub, and one page per pipeline.

    python report/build_report.py --dad-run outputs/dad/runs/<run_id>
    # -> report/index.html, report/dad.html

    python report/build_report.py --page dad --dad-run outputs/dad/runs/<run_id>

``--run`` still works as an alias for ``--dad-run``, which keeps the command printed in
the DAD page's own "Run it yourself" section true.

Each page is one self-contained HTML file: no external CSS, JS, fonts or images, so it
opens offline from the filesystem and survives an artifact host's CSP. The generators are
stdlib only and import nothing from viewer/ or shared/, so they build in an environment
where the pipeline's own dependencies are not installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report import common as C  # noqa: E402
from report import dad, hub  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
SHARED_CONTENT = REPORT_DIR / "content_shared.md"
DAD_CONTENT = REPORT_DIR / "content_dad.md"


def main():
    args = C.cli_parser(__doc__).parse_args()
    out_dir = Path(args.out_dir or REPORT_DIR)
    want = {"all": ("index", "dad"), "index": ("index",), "dad": ("dad",)}[args.page]

    if "dad" in want:
        if not args.dad_run:
            C.die("--dad-run is required to build the DAD page")
        kwargs = dad.load_inputs(args.dad_run, args.content or [DAD_CONTENT])
        audit = kwargs["audit"]
        html = dad.build(example=args.example,
                         sibling=("index.html", "Overview and the SDF corpus"), **kwargs)
        C.write(out_dir / "dad.html", html,
                label=f"n={audit.get('n_prompts')} "
                      f"delivery={'yes' if audit.get('delivery') else 'NO'} "
                      f"showcase={'yes' if audit.get('showcase') else 'NO'} "
                      f"diversity={'yes' if kwargs.get('diversity') else 'NO'}")

    if "index" in want:
        kwargs = hub.load_inputs(args.content or [SHARED_CONTENT], dad_run=args.dad_run,
                                 sdf_run=args.sdf_run)
        C.write(out_dir / "index.html", hub.build(**kwargs),
                label=f"dad={'yes' if kwargs.get('dad_audit') else 'NO'} "
                      f"sdf={'yes' if kwargs.get('sdf_audit') else 'NO'}")


if __name__ == "__main__":
    main()
