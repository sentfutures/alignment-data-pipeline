#!/usr/bin/env python3
"""The handoff page: both datasets, one file.

Written for one reader — someone who runs midtraining at another lab, has no context on
this project, and has about forty seconds to decide whether to keep reading. So the page
opens as one image and a sentence that runs straight on into what this is, puts the two
datasets side by side in a table they can read in one pass, and then asks them to choose
which one they want to read about.

Structure:

    hero        the illustration, the title, and #intro — what this is, in three lines
    #datasets   the two datasets, compared row by row
    #explore    Walk through either pipeline — two buttons
      #sdf      Synthetic documents  (report/sdf.py, hidden until chosen)
      #dad      Difficult advice     (report/dad.py, hidden until chosen)
    footer      repo, viewers, run ids, commits, build provenance

Nothing is open on load; ``#dad`` or ``#sdf`` in the URL opens that report, so the
dataset card's deep links land where they say they will, and printing expands both. The
cost of the chooser is that a closed report is invisible to Cmd-F — the tradeoff was
made deliberately, in favour of a reader who is choosing rather than scrolling.

stdlib only, and no imports from viewer/ or shared/.
"""

import datetime
import re

from report import common as C
from report import dad
from report import render as R
from report import sdf

CONTENT_IDS = ("title", "intro", "sdf_desc", "sdf_use", "sdf_unit",
               "dad_desc", "dad_use", "dad_unit")


REPO_URL = "https://github.com/sentfutures/alignment-data-pipeline"
# Deep links to the prompts each pipeline runs. The reader this page is written for
# wants the templates, not the records, so the comparison links straight at them.
PROMPTS_DAD = f"{REPO_URL}/tree/main/prompts/dad"
PROMPTS_SDF = f"{REPO_URL}/tree/main/prompts/sdf"
HF_URL = ("https://huggingface.co/datasets/sentientfutures/"
          "animal-welfare-mid-training-datasets")
HF_DAD = f"{HF_URL}/viewer/dad"
HF_SDF = f"{HF_URL}/viewer/sdf"

HERO_ALT = "A line drawing of a butterfly at the end of a looping dashed flight path."
# Inferred from the team's own domain; one constant to change if it is wrong.
MAKER, MAKER_URL = "Sentient Futures", "https://sentientfutures.ai"


def load_inputs(content_paths, dad_run=None, sdf_run=None):
    """All filesystem access, in one place. Returns build() kwargs."""
    ids = CONTENT_IDS + dad.CONTENT_IDS + sdf.CONTENT_IDS
    out = {"content": C.load_content(content_paths, ids)}
    if dad_run:
        out["dad_inputs"] = dad.load_inputs(dad_run)
    if sdf_run:
        out["sdf_inputs"] = sdf.load_inputs(sdf_run)
    return out


# ------------------------------------------------------------------ facts

# The page's own prose interpolates NOTHING: every figure on it is rendered by a section
# from a run's facts, so a {{placeholder}} in content_page.md is a build error. (The two
# model-name facts that used to live here existed only for the caveats strip.)
PAGE_FACTS = {}


def _date(manifest):
    """A run's generation date, in prose. Falls back to whatever the manifest holds."""
    raw = str((manifest or {}).get("created_at") or "")
    try:
        day = datetime.date.fromisoformat(raw[:10])
    except ValueError:
        return raw or "—"
    return f"{day.day} {day:%B %Y}"


# ------------------------------------------------------------------ sections

def section_datasets(content, f, dad_kwargs, sdf_kwargs):
    """The two datasets, side by side. Their names are this section's heading.

    Five rows: three describe the RESULT — what the dataset is, what one record is, what
    it is for — before the two that link out to the templates and to a made example.

    A ``pipeline`` row naming each chain of stages used to sit between them. It was cut:
    the two walkthroughs below ARE the pipeline, at length and with a diagram each, and a
    one-line chain above them was a summary the reader met before it could mean anything.

    ``result`` was the masthead's subtitle. It reads as a row because it is one: the
    sentence saying what each dataset is was the only unlabelled claim in the comparison.

    How MANY records is not a row — that is a property of a run, and this section
    describes the pipelines; the counts live in each report's appendix, beside the run
    they came off. Dates, model ids and the composition spread belong there too, not in a
    table meant to be read in one pass.
    """
    rows = [
        ("result", _cell(content, "sdf_desc", f), _cell(content, "dad_desc", f)),
        ("result format", _cell(content, "sdf_unit", f), _cell(content, "dad_unit", f)),
        ("what it is for", _cell(content, "sdf_use", f), _cell(content, "dad_use", f)),
        ("prompt templates", _prompts_cell(sdf_kwargs, PROMPTS_SDF),
         _prompts_cell(dad_kwargs, PROMPTS_DAD)),
        ("example dataset",
         _with_button("" if sdf_kwargs else "not published yet", HF_SDF,
                      "Example dataset", "hf"),
         _with_button("", HF_DAD, "Example dataset", "hf")),
    ]
    columns = [(sdf.SECTION_TITLE,), (dad.SECTION_TITLE,)]
    # The heading is heard, not seen: the two mastheads are the heading on screen.
    return C.section("datasets", "The two datasets", R.compare(columns, rows),
                     heading_class="vh")


def _prompts_cell(kwargs, href):
    """How many prompt templates the pipeline is — the figure a reader who wants to run
    it against their own model is after, rather than how much data we happened to make.
    Counted from the run's own inputs/prompts snapshot; see common.prompt_count."""
    n = (kwargs or {}).get("n_prompt_templates")
    return _with_button(f"{n}" if n else "—", href, "Templates", "github")


def _with_button(value, href, label, icon):
    """A cell's figure at the left of its column, and the way to the thing it counts at
    the right. An empty value still leaves its flex item behind, so a button-only cell
    lines its button up under the one in the row above."""
    return R.Raw(f"<span class='cmp-fig'><span>{R.esc(value)}</span>"
                 f"{R.linkbutton(href, label, icon)}</span>")



def _cell(content, key, f):
    return R.Raw(R.inline_md(C.fill(content.get(key, ""), f)))






def section_explore(panels, outlines):
    """The choice, both reports under it, and each one's contents beside it.

    Two names on the buttons, nothing else: what each dataset is and how big it is are in
    the comparison directly above, and repeating them here only made the buttons hard to
    read as buttons. The report's own beats and stages go in the rail beside it, read back
    off the panel that was built, so a rail link cannot name a beat the report did not
    render.

    The panels are nested here rather than left as siblings in ``<main>`` because the
    buttons and the rail stay on screen while a report is read, and a sticky box travels
    only inside its containing block — see ``render.explore_body``.
    """
    rails = "".join(R.rail(pid, outlines.get(pid, ())) for pid in ("sdf", "dad"))
    return C.section("explore", "Walk through either pipeline",
                     R.explore_body(
                         R.chooser([("sdf", sdf.SECTION_TITLE),
                                    ("dad", dad.SECTION_TITLE)]),
                         rails, panels))


def footer(maker_icon=""):
    """Who made it and where to go.

    No run ids, no commits, no backend. The page reads as the pipeline running now rather
    than as a report on one batch that finished, and a footer naming two run directories
    dates it to that batch. What a reader needs to locate the data is the two links, and
    what they need to reproduce it is the repository.
    """
    mark = (f"<img class='ico-img' src='{R.esc(maker_icon)}' alt=''>" if maker_icon else "")
    return (f"<p>A project by <a href='{MAKER_URL}'{R.NEW_TAB}>{mark}{R.esc(MAKER)}"
            f"{R.EXT_ARROW}</a></p>"
            f"<p class='foot-links'>{R.iconlink(HF_URL, 'Datasets', 'hf')}"
            f"{R.iconlink(REPO_URL, 'Pipelines', 'github')}</p>")


# ------------------------------------------------------------------ assembly

def body(*, content, dad_inputs=None, sdf_inputs=None, example=None, sdf_example=None,
         illustration="", maker_icon=""):
    """The masthead and the sections. Pure: no filesystem, no argv."""
    dad_kwargs, sdf_kwargs = dad_inputs or {}, sdf_inputs or {}
    f = dict(PAGE_FACTS)
    title = C.fill(content["title"], f).strip()

    # Synthetic documents first, throughout: the comparison, the chooser and the panels
    # all read in one order. Each report's contents come back off its own built markup, so
    # the rail is the outline of the report that was actually rendered.
    bodies = [(sdf.SECTION_ID,
               f"<h2>{R.esc(sdf.SECTION_TITLE)}</h2>"
               + sdf.blocks(content=content, example=sdf_example, hf_href=HF_SDF,
                            repo_href=REPO_URL, **sdf_kwargs))]
    if dad_kwargs:
        bodies.append((dad.SECTION_ID,
                       f"<h2>{R.esc(dad.SECTION_TITLE)}</h2>"
                       + dad.blocks(content=content, example=example,
                                    hf_href=HF_DAD, repo_href=REPO_URL, **dad_kwargs)))
    panels = "".join(R.panel(pid, html) for pid, html in bodies)
    outlines = {pid: R.outline(html) for pid, html in bodies}
    sections = [
        section_datasets(content, f, dad_kwargs, sdf_kwargs),
        section_explore(panels, outlines),
    ]
    head = {
        "title": title,
        "masthead": R.hero(title, R.illustration(illustration, alt=HERO_ALT),
                           intro=C.prose(content, "intro", f)),
        "footer": footer(maker_icon),
    }
    return "".join(sections), head


def build(**kwargs):
    body_html, head = body(**kwargs)
    return R.document(body=body_html, **head)
