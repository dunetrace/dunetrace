from __future__ import annotations

import os
import unittest
from importlib import reload
from unittest.mock import patch

import semantic_svc.config as cfg_mod


class TestSemanticWorkerEnabledFlag(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEMANTIC_WORKER_ENABLED", None)
            reload(cfg_mod)
            self.assertFalse(cfg_mod.Settings().SEMANTIC_WORKER_ENABLED)
        reload(cfg_mod)

    def test_enabled_via_env_true(self):
        with patch.dict(os.environ, {"SEMANTIC_WORKER_ENABLED": "true"}):
            reload(cfg_mod)
            self.assertTrue(cfg_mod.Settings().SEMANTIC_WORKER_ENABLED)
        reload(cfg_mod)  # restore to original env values (outside patch.dict)

    def test_enabled_via_env_1(self):
        with patch.dict(os.environ, {"SEMANTIC_WORKER_ENABLED": "1"}):
            reload(cfg_mod)
            self.assertTrue(cfg_mod.Settings().SEMANTIC_WORKER_ENABLED)
        reload(cfg_mod)

    def test_falsy_string_stays_disabled(self):
        with patch.dict(os.environ, {"SEMANTIC_WORKER_ENABLED": "false"}):
            reload(cfg_mod)
            self.assertFalse(cfg_mod.Settings().SEMANTIC_WORKER_ENABLED)
        reload(cfg_mod)


if __name__ == "__main__":
    unittest.main()
