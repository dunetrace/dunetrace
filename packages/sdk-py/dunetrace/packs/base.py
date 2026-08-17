"""
Pack registration (Phase 1.0). A DetectorPack is a Dunetrace-owned bundle of
detector classes a customer activates as a whole — distinct from
dunetrace.detectors.CUSTOM_DETECTOR_REGISTRY, which is for a customer's own
detector logic dropped into ~/.dunetrace/detectors/.

Registration happens at import time (register_pack() called at module scope
in each pack module, e.g. packs/voice.py) — the same "importing the module is
what makes it known" idiom BaseDetector.__init_subclass__ already uses for
custom detectors, just keyed by pack name instead of class name.

PACK_REGISTRY only records which detector classes belong to which pack. It
says nothing about which orgs have activated a pack — that's
org_enabled_packs, read by detector_svc/packs.py. A pack's classes exist in
this registry (and get instantiated when detector_svc seeds the packs table)
regardless of whether any org has activated it; activation only controls
whether get_detectors() includes its instances in a given org's evaluation
list.
"""

from __future__ import annotations

from typing import Dict, List, Type

from dunetrace.detectors import BaseDetector


class DetectorPack:
    name: str
    description: str
    detectors: List[Type[BaseDetector]]


PACK_REGISTRY: Dict[str, DetectorPack] = {}


def register_pack(pack: DetectorPack) -> None:
    """Registers a pack. Last-write-wins on a name collision — same
    "no additional surprise" reasoning CUSTOM_DETECTOR_REGISTRY already
    applies to a duplicate class name."""
    PACK_REGISTRY[pack.name] = pack
