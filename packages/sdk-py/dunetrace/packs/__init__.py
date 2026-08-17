# First-party detector packs (Phase 1.0 infrastructure). Each pack module
# registers itself via register_pack() at import time, so importing this
# package is what makes every pack's detector classes known to
# PACK_REGISTRY — add a pack by importing its module below, not by editing
# any other file.

from dunetrace.packs.base import DetectorPack, PACK_REGISTRY, register_pack

# Importing the module runs its register_pack() call (Phase 1.2 voice pack).
from dunetrace.packs import voice  # noqa: E402,F401

__all__ = ["DetectorPack", "PACK_REGISTRY", "register_pack"]
