#!/usr/bin/env python3
"""
Tests for check_endpoint_conventions.py itself — a static-analysis tool is
only useful if its detection and suppression logic are actually correct, so
this isn't left untested just because it lives in scripts/ rather than a
service's own test suite. Self-contained: writes real files into a temp
directory shaped like services/<x>/<y>/routers/*.py, no fixtures needed
from the rest of the repo.

Run: python scripts/test_check_endpoint_conventions.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_endpoint_conventions import check_file, find_router_files


def _write_router(root: Path, service: str, content: str) -> Path:
    router_dir = root / "services" / service / f"{service}_svc" / "routers"
    router_dir.mkdir(parents=True, exist_ok=True)
    path = router_dir / "example.py"
    path.write_text(content)
    return path


class TestFindRouterFiles(unittest.TestCase):
    def test_finds_files_matching_the_glob_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_router(root, "widgets", "# empty\n")
            found = find_router_files(root)
        self.assertEqual(len(found), 1)
        self.assertTrue(str(found[0]).endswith("routers/example.py"))

    def test_ignores_files_outside_the_routers_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "services" / "widgets" / "widgets_svc" / "other.py"
            other.parent.mkdir(parents=True)
            other.write_text("@router.get('/v1/orgs/{org_id}/x')\n")
            found = find_router_files(root)
        self.assertEqual(found, [])


class TestCheckFile(unittest.TestCase):
    def _check(self, content: str) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_router(Path(tmp), "widgets", content)
            return check_file(path)

    def test_org_id_in_path_is_flagged(self):
        violations = self._check(
            '@router.post("/v1/orgs/{org_id}/packs/{pack_name}")\n'
            "async def activate(org_id: str, pack_name: str):\n    pass\n"
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][1], "POST")
        self.assertIn("{org_id}", violations[0][2])

    def test_org_id_derived_from_require_org_is_not_flagged(self):
        violations = self._check(
            '@router.post("/v1/orgs/packs/{pack_name}")\n'
            "async def activate(pack_name: str, org_id: str = Depends(require_org)):\n    pass\n"
        )
        self.assertEqual(violations, [])

    def test_suppression_marker_immediately_above_silences_it(self):
        violations = self._check(
            "# org-id-path-ok: third-party webhook, no Dunetrace auth on this request\n"
            '@router.post("/v1/webhooks/linear/{org_id}")\n'
            "async def linear_webhook(org_id: str):\n    pass\n"
        )
        self.assertEqual(violations, [])

    def test_suppression_marker_several_lines_above_still_silences_it(self):
        violations = self._check(
            "# org-id-path-ok: third-party webhook, no Dunetrace auth on this request\n"
            "# (multi-line explanation continues here)\n"
            "# and here too\n"
            '@router.post("/v1/webhooks/linear/{org_id}")\n'
            "async def linear_webhook(org_id: str):\n    pass\n"
        )
        self.assertEqual(violations, [])

    def test_suppression_marker_attached_to_a_different_endpoint_does_not_leak_forward(self):
        """A suppression comment belongs to the ONE decorator it's
        contiguously attached to — real code (another endpoint's body) in
        between breaks the chain, so it must not silence a later,
        unrelated endpoint just because both happen to live in the same
        file."""
        violations = self._check(
            "# org-id-path-ok: this marker belongs to the endpoint directly below\n"
            '@router.post("/v1/webhooks/x/{org_id}")\n'
            "async def hook(org_id: str):\n"
            "    pass\n"
            "\n"
            '@router.post("/v1/orgs/{org_id}/other")\n'
            "async def other(org_id: str):\n    pass\n"
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][2], "/v1/orgs/{org_id}/other")

    def test_multiple_endpoints_in_one_file_each_checked_independently(self):
        violations = self._check(
            '@router.get("/v1/orgs/packs")\n'
            "async def list_packs(org_id: str = Depends(require_org)):\n    pass\n\n"
            "# org-id-path-ok: webhook, no auth\n"
            '@router.post("/v1/webhooks/x/{org_id}")\n'
            "async def hook(org_id: str):\n    pass\n\n"
            '@router.delete("/v1/orgs/{org_id}/bad")\n'
            "async def bad(org_id: str):\n    pass\n"
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][2], "/v1/orgs/{org_id}/bad")

    def test_no_endpoints_returns_empty_list(self):
        violations = self._check("# just a comment, no routes here\n")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
