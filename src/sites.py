"""Choosing *which* ``(layer, head)`` sites an intervention acts on.

:mod:`scripts.write` crosses a site count with a selection rule: at each count the same
number of sites is edited, chosen by what they carry (``top``), at random (``random``) or
by carrying the least (``bottom``), so a difference between the three rows at one count
is attributable to the selection and nothing else. A head index would be another way to
pick 32 sites -- ``(4, 14)`` and ``(25, 14)`` as one object, on the strength of a shared
slot in a tensor -- but a head index is not a function, and the experiment asks about
sites selected by what they carry, at every count.

The score is the workspace loading of :mod:`src.structural`, unchanged and reused; nothing
here introduces new algebra, and nothing here touches a model.

**Selection is on the source concept**, which the algebra forces. The swap's update is
``alpha (c_t - c_s)(v_s - v_t)``, so a site carrying none of ``s`` has nothing to exchange
and the update there is driven entirely by ``c_t`` -- the concept the model is *not*
currently expressing.

**The threshold rule needs a per-site null.** Loading scales differ across sites, so a
single global cut would select whichever sites happen to have large cosines with everything.
:func:`null_threshold` takes the loading of an arbitrary concept population at each site and
returns that site's own quantile, which turns "the concept is active here" into "more
active here than an arbitrary concept is".
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from src.heads import COMPUTE_DTYPE
from src.structural import workspace_loading

logger = logging.getLogger(__name__)

#: A single intervention site: ``(layer, head)``.
Site = tuple[int, int]

#: Selection rules :func:`select_sites` accepts.
RULES = ("top", "bottom", "random", "threshold")


def loading_map(
    jz_by_layer: dict[int, torch.Tensor],
    z_by_layer: dict[int, torch.Tensor],
    spec,
    unembed_rows: torch.Tensor,
    layers: list[int],
) -> torch.Tensor:
    """``cos(z_{l,h}, v_{s,l,h})`` over a set of layers, as ``[n_layers, n_heads]``.

    :func:`src.structural.workspace_loading` swept over layers. ``layers`` fixes the row
    order explicitly rather than relying on dict insertion order, because every downstream
    index into this tensor is a ``(layer, head)`` pair that has to mean the same thing in
    the selection, the edit and the JSON record.

    Args:
        jz_by_layer: ``{layer: J^z_l}``, each ``[d_model, d_source]``.
        z_by_layer: ``{layer: z_l}``, each ``[d_source]`` -- one position's ``o_proj``
            input, at the item's documented readout position.
        unembed_rows: ``[d_model]`` for the concept, or ``[n_tokens, d_model]`` for a
            population (the threshold null). Must already carry the final norm's gain.

    Returns:
        ``[n_layers, n_heads]``, or ``[n_tokens, n_layers, n_heads]`` for a population.
    """
    missing = [l for l in layers if l not in jz_by_layer or l not in z_by_layer]
    if missing:
        raise ValueError(f"no J^z or z for layers {missing}")

    per_layer = [
        workspace_loading(
            jz_by_layer[layer], z_by_layer[layer], spec, unembed_rows
        )
        for layer in layers
    ]
    # [n_heads] per layer -> [n_layers, n_heads]; [n_tokens, n_heads] -> [n_tokens, n_layers, n_heads].
    return torch.stack(per_layer, dim=-2)


def null_threshold(null_loading: torch.Tensor, *, q: float = 0.99) -> torch.Tensor:
    """Per-site loading a concept must beat to count as active, ``[n_layers, n_heads]``.

    Args:
        null_loading: ``[n_tokens, n_layers, n_heads]`` from :func:`loading_map` over an
            arbitrary concept population.
        q: Quantile. At the default 0.99 a site admits an arbitrary concept 1% of the
            time, so a 1024-site grid selects ~10 sites by chance and the rule is
            genuinely selective.

    The quantile is taken **per site** rather than pooled. Pooling would let sites whose
    lens vectors happen to align with much of the vocabulary dominate the cut, and those
    are exactly the sites a concept-selection experiment must not be handed for free.
    """
    if null_loading.ndim != 3:
        raise ValueError(
            f"expected [n_tokens, n_layers, n_heads] null, got {tuple(null_loading.shape)}"
        )
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    return torch.quantile(null_loading.float(), q, dim=0)


def select_sites(
    loading: torch.Tensor,
    layers: list[int],
    rule: str,
    *,
    k: int | None = None,
    tau: torch.Tensor | float | None = None,
    seed: int | None = None,
) -> list[Site]:
    """The site set one condition edits, as ``(layer, head)`` pairs.

    Returned **sorted by (layer, head)**, not by score: the set is the object under test,
    and a stable order keeps the JSON record diffable and the per-layer edit grouping
    trivial. Look scores up in ``loading`` if the ordering is wanted.

    Ties are broken by flattened index, so a degenerate loading map still yields a
    deterministic set rather than one that depends on the sort's stability.

    Args:
        loading: ``[n_layers, n_heads]`` from :func:`loading_map`.
        layers: The layer index each row stands for.
        rule: One of :data:`RULES`. ``top`` / ``bottom`` / ``random`` need ``k``,
            ``threshold`` needs ``tau``, ``random`` needs ``seed``.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    if loading.ndim != 2:
        raise ValueError(f"expected [n_layers, n_heads] loading, got {tuple(loading.shape)}")
    n_layers, n_heads = loading.shape
    if len(layers) != n_layers:
        raise ValueError(f"{len(layers)} layers for {n_layers} loading rows")

    def sites_from(flat_indices) -> list[Site]:
        pairs = {
            (layers[int(i) // n_heads], int(i) % n_heads) for i in flat_indices
        }
        return sorted(pairs)

    if rule == "threshold":
        if tau is None:
            raise ValueError("the threshold rule needs tau")
        cut = tau if torch.is_tensor(tau) else torch.full_like(loading, float(tau))
        if cut.shape != loading.shape:
            raise ValueError(f"tau is {tuple(cut.shape)}, expected {tuple(loading.shape)}")
        mask = (loading > cut.to(loading.dtype)).reshape(-1)
        return sites_from(torch.nonzero(mask).flatten())

    n_sites = n_layers * n_heads
    if k is None:
        raise ValueError(f"the {rule} rule needs k")
    if not 0 < k <= n_sites:
        raise ValueError(f"k={k} out of range for {n_sites} sites")

    if rule == "random":
        if seed is None:
            raise ValueError("the random rule needs a seed, or the control is not reproducible")
        generator = torch.Generator().manual_seed(seed)
        return sites_from(torch.randperm(n_sites, generator=generator)[:k])

    flat = loading.reshape(-1)
    largest = rule == "top"
    return sites_from(flat.topk(k, largest=largest).indices)


def plan_edits(
    elements: list[tuple[list[Site], str]],
) -> dict[int, dict[str, dict[int, list[int]]]]:
    """Invert ``[(sites, arm)]`` into ``{layer: {arm: {head: [batch rows]}}}``.

    The causal driver batches over ``(condition, arm)`` pairs, so a forward pass carries a different
    site set in every batch element. Inverting once, before the pass, is what keeps the
    hook's cost proportional to the sites actually touched rather than to
    ``elements x heads``: each ``(layer, arm, head)`` is handled in a single slice serving
    every element that selected it.

    Grouping by arm inside the layer is not cosmetic -- the three arms swap along
    different directions, so they need different frames and cannot share a call.
    """
    plan: dict[int, dict[str, dict[int, list[int]]]] = {}
    for index, (sites, arm) in enumerate(elements):
        for layer, head in sites:
            plan.setdefault(layer, {}).setdefault(arm, {}).setdefault(head, []).append(index)
    return plan


def apply_site_deltas(
    z: torch.Tensor,
    spec,
    plan_for_layer: dict[str, dict[int, list[int]]],
    delta_for: Callable[[str, int, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    totals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add per-site updates to one layer's ``z``, batched over elements.

    The tensor mechanics of the site-selected edit, kept here rather than in the driver so
    they can be tested: ``z`` is ``[n_elements, seq, d_source]`` and only the head slices
    a given element selected may move. Updates accumulate into a zero tensor and are added
    once, so an element that selected no site at this layer is returned bit-identical
    rather than merely close -- an edit that silently perturbed the unselected elements
    would quietly turn every condition into the same condition.

    Args:
        plan_for_layer: ``{arm: {head: [batch rows]}}`` from :func:`plan_edits`.
        delta_for: ``(arm, head, block, index) -> delta``, where ``block`` is
            ``[len(rows), seq, head_dim]``. Supplies the swap semantics; this function
            supplies only the indexing.
        totals: Optional ``[n_elements]`` accumulator of squared update norms, summed
            across layers. The norm-equalised controls need it, and it is the audit
            that says whether the equalisation held.
    """
    delta_full = torch.zeros_like(z, dtype=COMPUTE_DTYPE)
    for arm, by_head in plan_for_layer.items():
        for head, rows in by_head.items():
            index = torch.tensor(rows, device=z.device)
            head_slice = spec.head_slice(head)
            block = z[index][:, :, head_slice]
            delta = delta_for(arm, head, block, index).to(COMPUTE_DTYPE)
            if delta.shape != block.shape:
                raise ValueError(
                    f"delta for arm {arm!r} head {head} is {tuple(delta.shape)}, "
                    f"expected {tuple(block.shape)}"
                )
            if totals is not None:
                totals.index_add_(
                    0,
                    index.to(totals.device),
                    delta.detach().float().pow(2).sum(dim=(1, 2)).to(totals.device),
                )
            # Basic slicing gives a view, so this writes through into delta_full.
            delta_full[:, :, head_slice].index_add_(0, index, delta)
    return (z.to(COMPUTE_DTYPE) + delta_full).to(z.dtype)


def site_spread(sites: list[Site], *, n_heads: int, n_kv_heads: int) -> dict[str, int]:
    """How independent a site set actually is, for the record.

    Llama-3.1-8B is GQA: one ``W_V`` is shared across each group of ``n_heads /
    n_kv_heads`` query heads, so two selected sites in the same layer and group have
    partially shared OV circuits. A ``top11`` that keeps picking group-mates is not eleven
    independent sites, and the same caution applies to a set concentrated in one layer.

    Recorded per trial rather than assumed away, because both failure modes are invisible
    in a success count.
    """
    if n_heads % n_kv_heads:
        raise ValueError(f"{n_heads} query heads do not divide into {n_kv_heads} groups")
    per_group = n_heads // n_kv_heads
    layer_groups = [(layer, head // per_group) for layer, head in sites]
    counts: dict[tuple[int, int], int] = {}
    for key in layer_groups:
        counts[key] = counts.get(key, 0) + 1
    return {
        "n_sites": len(sites),
        "n_layers": len({layer for layer, _ in sites}),
        "n_layer_groups": len(counts),
        "max_sites_per_layer_group": max(counts.values()) if counts else 0,
    }
