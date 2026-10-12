"""Everything ``notebooks/head_lens_explorer.py`` needs that is not a marimo cell.

The notebook is exploration and presentation; no number in it is a
result. These modules keep that split visible: :mod:`~explorer.session`,
:mod:`~explorer.readings` and :mod:`~explorer.editing` only *call* ``src`` and the
drivers' protocol constants, while :mod:`~explorer.charts` never touches the model -- it
takes arrays and token strings, so a colour can change without any risk of moving a
number. :mod:`~explorer.records` reads the results the paper cites, so the notebook can
check its live numbers against them rather than ask to be trusted.

===========================  =========================================================
Module                       Concern
===========================  =========================================================
:mod:`~explorer.session`     Load the model and both lenses once; prove they are the
                             bytes ``results/pilot/`` was computed on.
:mod:`~explorer.prompts`     The prompt menu, the readout position, the candidate rule.
:mod:`~explorer.readings`    One forward pass -> both head readouts and J-layer at one
                             position, computed as ``scripts/read.py`` computes them.
:mod:`~explorer.editing`     The coordinate swap at head sites, J-head and Logit-head,
                             computed as ``scripts/write_swap_sites.py`` computes it.
:mod:`~explorer.records`     The records under ``results/``: the paper's tables and the
                             per-item / per-trial values a live run is checked against.
:mod:`~explorer.charts`      Every chart, as an Altair spec built from plain arrays.
:mod:`~explorer.saving`      Write the current readings, edits and charts out with a
                             provenance block.
===========================  =========================================================

The package is called ``explorer`` rather than ``scripts`` so that ``scripts`` keeps
meaning the repository's drivers. Marimo puts only ``notebooks/`` on ``sys.path``; the
repository root is appended below so that :mod:`scripts.write`'s protocol constants
(site counts, null seed and quantile, random-site seed, alphas) and its ``site_sets`` are
*imported* by :mod:`~explorer.editing`, never copied. Nothing here touches a GPU on
import.

Marimo does not reload an edited module by default: turn on module autoreloading in the
notebook settings, or re-run the import cell, after changing a file here.
"""

import sys

from src.config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
