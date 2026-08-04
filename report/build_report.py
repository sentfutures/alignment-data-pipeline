#!/usr/bin/env python3
"""Build the handoff page: both corpora, one self-contained HTML file.

    python report/build_report.py --dad-run outputs/dad/runs/<run_id> \\
                                  --sdf-run outputs/sdf/runs/<run_id>
    # -> report/index.html

``--run`` still works as an alias for ``--dad-run``, which keeps the command printed in
the page's own "Running it yourself" block true. ``--sdf-run`` is optional: without it
the document corpus's column and section say so instead of showing figures.

The page is one self-contained HTML file: no external CSS, JS, fonts or images, so it
opens offline from the filesystem, survives an artifact host's CSP, and publishes to
GitHub Pages as-is. The generator is stdlib only and imports nothing from viewer/ or
shared/, so it builds in an environment where the pipeline's own dependencies are not
installed.

For a HOSTED build, name the URL it will be served from — that adds the link-preview tags
and copies the card image out beside the page, the one file that travels with it:

    python report/build_report.py --dad-run <run> --sdf-run <run> \\
                                  --site-url https://<host>/
    # -> report/index.html + report/preview.png

Every build is `noindex` either way; see "Hosting" in report/README.md.
"""

import base64
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report import common as C  # noqa: E402
from report import page  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
CONTENT = [REPORT_DIR / "content_page.md", REPORT_DIR / "content_dad.md",
           REPORT_DIR / "content_sdf.md"]
HERO = REPORT_DIR / "assets" / "hero.png"
MAKER_ICON = REPORT_DIR / "assets" / "sf.png"
# The link preview's image — the hero on the page's own paper, drawn by make_preview.py.
# The ONE file that travels beside index.html, and only for a hosted build: a card renderer
# fetches og:image over the network, so this is the one picture the page cannot carry.
PREVIEW = REPORT_DIR / "assets" / "preview.png"


def data_uri(path, mime="image/png"):
    """An image as a data: URI, or "" if it is not there.

    Inlining is not an optimisation here, it is the format: the page has to be one file
    that opens offline, so the only picture it can carry is one encoded into it. The
    source art lives in report/assets/ and never ships next to the HTML.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ""
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def main():
    args = C.cli_parser(__doc__).parse_args()
    if not args.dad_run:
        C.die("--dad-run is required")
    out_dir = Path(args.out_dir or REPORT_DIR)
    kwargs = page.load_inputs(args.content or CONTENT, dad_run=args.dad_run,
                              sdf_run=args.sdf_run)
    hero = data_uri(HERO)
    site_url = args.site_url or ""
    # A hosted build gets the card image without being asked twice: the default names the
    # file this script copies out beside index.html, resolved against the page's own URL
    # (urljoin, so both ".../" and ".../index.html" land on ".../preview.png").
    preview_url = args.preview_url or (urljoin(site_url, PREVIEW.name) if site_url else "")
    html = page.build(example=args.example, sdf_example=args.sdf_example, illustration=hero,
                      maker_icon=data_uri(MAKER_ICON), site_url=site_url,
                      preview_url=preview_url, **kwargs)
    if site_url and not args.preview_url and PREVIEW.exists():
        shutil.copyfile(PREVIEW, out_dir / PREVIEW.name)
    audit = (kwargs.get("dad_inputs") or {}).get("audit") or {}
    C.write(out_dir / "index.html", html,
            label=f"{C.editorial_words(html):,} words of prose · "
                  f"hero={'inlined' if hero else 'NO'} · "
                  f"dad n={audit.get('n_prompts')} "
                  f"delivery={'yes' if audit.get('delivery') else 'NO'} "
                  f"showcase={'yes' if audit.get('showcase') else 'NO'} "
                  f"sdf={'yes' if kwargs.get('sdf_inputs') else 'NO'} "
                  f"preview={preview_url or 'no'}")


if __name__ == "__main__":
    main()
