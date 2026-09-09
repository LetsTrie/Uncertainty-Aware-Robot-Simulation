"""Turn a chosen target object into a natural-language instruction.

The instruction may be fully specified ("put the blue mug beside the laptop"),
category-only ("put the mug ..."), or generic ("put the object ..."). Under-
specification is exactly what creates *referential ambiguity* — the signal the
ASK decision is meant to resolve.
"""
from __future__ import annotations

from ..types import Instruction, Obj


def make_instruction(target: Obj, landmark: str, mode_probs, rng) -> Instruction:
    """Build an instruction referring to ``target``.

    ``mode_probs`` is (p_full, p_cat_only, p_generic); they should sum to ~1.
    """
    p_full, p_cat_only, p_generic = mode_probs
    r = rng.random()
    if r < p_full:
        ref_color, ref_category = target.color, target.category
    elif r < p_full + p_cat_only:
        ref_color, ref_category = None, target.category
    else:
        ref_color, ref_category = None, None

    if ref_category is None:
        noun = "object"
    elif ref_color is None:
        noun = ref_category
    else:
        noun = f"{ref_color} {ref_category}"

    text = f"Put the {noun} beside the {landmark}."
    return Instruction(text=text, ref_category=ref_category,
                       ref_color=ref_color, landmark=landmark)


def matching_ids(instruction: Instruction, objects) -> list[str]:
    """Object ids consistent with the referring expression."""
    out = []
    for o in objects:
        if instruction.ref_category is not None and o.category != instruction.ref_category:
            continue
        if instruction.ref_color is not None and o.color != instruction.ref_color:
            continue
        out.append(o.id)
    return out
