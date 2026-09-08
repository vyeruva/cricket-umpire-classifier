"""Shared filename -> label logic.

Filenames look like: no_ball_003.jpg, out_012.jpg, sixes_045.jpg, wide_007.jpg,
no_action_099.jpg

A naive "prefix before the first underscore" split breaks here: both
no_ball_* and no_action_* start with "no_", so split('_')[0] gives "no" for
both instead of the real two-word class name. Match against the known class
prefixes instead, longest/most-specific first.
"""

import re
from pathlib import Path

# Order matters: no_ball/no_action must be checked before a bare "no" would
# ever match anything else. Kept as an ordered list, not a set.
CLASS_PREFIXES = ["no_ball", "no_action", "sixes", "out", "wide"]

# Canonical display names for the 5 classes in the task spec.
DISPLAY_NAME = {
    "sixes": "Six",
    "no_ball": "No ball",
    "out": "Out",
    "wide": "Wide",
    "no_action": "No action",
}

LABEL_RE = re.compile(r"^(" + "|".join(CLASS_PREFIXES) + r")_")


def label_from_filename(path: "str | Path") -> str:
    name = Path(path).name
    match = LABEL_RE.match(name)
    if not match:
        raise ValueError(f"Could not infer class label from filename: {name}")
    return DISPLAY_NAME[match.group(1)]
