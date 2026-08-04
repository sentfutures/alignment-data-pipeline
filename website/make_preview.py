#!/usr/bin/env python3
"""Draw the link-preview image from the hero: website/assets/preview.png.

    python website/make_preview.py        # -> website/assets/preview.png (1200x630)

Run it when `assets/hero.png` changes, and commit the result. It is NOT part of
`build_website.py`, which is stdlib-only on purpose — this needs Pillow, and the preview is
an asset, not a build product.

The card image cannot be carried inside the page. `og:image` is fetched out of band by
whoever renders the link, so a data URI is not an option: this file is uploaded beside
index.html and named by `--preview-url` (see "Hosting" in website/README.md).

What it does to the art: nothing. The hero is trimmed to its own alpha bounds, scaled to
fit, and centred on the page's paper — no crop through the drawing, no filter, no text
baked over it. 1200x630 is what the card consumers expect at 2:1-ish; the hero is 3:2, so
fitting it whole leaves paper either side, which is what the page does with it too.
"""

from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets"
SRC, OUT = ASSETS / "hero.png", ASSETS / "preview.png"
SIZE = (1200, 630)
PAPER = (247, 244, 234)          # --surface-0, the page's own paper
MARGIN = 0.86                    # of the shorter axis, so the art is not flush to the edge


def main():
    art = Image.open(SRC).convert("RGBA")
    art = art.crop(art.getbbox() or (0, 0, *art.size))    # its own alpha bounds
    scale = min(SIZE[0] * MARGIN / art.width, SIZE[1] * MARGIN / art.height)
    art = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)
    card = Image.new("RGB", SIZE, PAPER)
    card.paste(art, ((SIZE[0] - art.width) // 2, (SIZE[1] - art.height) // 2), art)
    card.save(OUT, optimize=True)
    print(f"wrote {OUT} ({SIZE[0]}x{SIZE[1]}, {OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
