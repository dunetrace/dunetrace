"""
Tests for A3: Python-class custom detector registration parity.

Covers custom_python_detectors.py's plugin loading (isolation, error handling,
path resolution) and detectors.py's merging of registered-but-unlisted
classes into get_detectors(). CUSTOM_DETECTOR_REGISTRY is shared module-level
state in dunetrace.detectors — every test here patches it to a fresh dict.

Run:
    PYTHONPATH=packages/sdk-py:services/detector pytest services/detector/tests/test_custom_python_detectors.py -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import dunetrace.detectors as detectors_module
from dunetrace.detectors import BaseDetector

from detector_svc.custom_python_detectors import load_custom_detector_plugins
import detector_svc.detectors as detector_svc_detectors


class _RegistryIsolatedTestCase(unittest.TestCase):
    def setUp(self):
        self._patcher = mock.patch.dict(detectors_module.CUSTOM_DETECTOR_REGISTRY, clear=True)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()


class TestLoadCustomDetectorPlugins(_RegistryIsolatedTestCase):
    def test_missing_directory_returns_zero_not_an_error(self):
        loaded = load_custom_detector_plugins("/nonexistent-plugin-dir-xyz")
        self.assertEqual(loaded, 0)

    def test_valid_plugin_file_is_loaded_and_registers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "my_plugin.py"), "w") as f:
                f.write(
                    "from dunetrace.detectors import BaseDetector\n"
                    "class MyLoadedPlugin(BaseDetector):\n"
                    "    name = 'MY_LOADED_PLUGIN'\n"
                )
            loaded = load_custom_detector_plugins(tmp)
        self.assertEqual(loaded, 1)
        self.assertIn("MyLoadedPlugin", detectors_module.CUSTOM_DETECTOR_REGISTRY)

    def test_malformed_plugin_file_is_skipped_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "broken_plugin.py"), "w") as f:
                f.write("this is not valid python syntax :::\n")
            loaded = load_custom_detector_plugins(tmp)  # must not raise
        self.assertEqual(loaded, 0)

    def test_one_malformed_file_does_not_block_a_valid_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a_broken.py"), "w") as f:
                f.write("raise RuntimeError('boom at import time')\n")
            with open(os.path.join(tmp, "b_valid.py"), "w") as f:
                f.write(
                    "from dunetrace.detectors import BaseDetector\n"
                    "class ValidSibling(BaseDetector):\n"
                    "    name = 'VALID_SIBLING'\n"
                )
            loaded = load_custom_detector_plugins(tmp)
        self.assertEqual(loaded, 1)
        self.assertIn("ValidSibling", detectors_module.CUSTOM_DETECTOR_REGISTRY)

    def test_non_py_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "readme.txt"), "w") as f:
                f.write("not python")
            loaded = load_custom_detector_plugins(tmp)
        self.assertEqual(loaded, 0)

    def test_files_starting_with_underscore_are_ignored(self):
        # __init__.py, __pycache__ artifacts, etc. — not plugin entry points.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "_private.py"), "w") as f:
                f.write(
                    "from dunetrace.detectors import BaseDetector\n"
                    "class ShouldNotLoad(BaseDetector):\n"
                    "    name = 'SHOULD_NOT_LOAD'\n"
                )
            loaded = load_custom_detector_plugins(tmp)
        self.assertEqual(loaded, 0)
        self.assertNotIn("ShouldNotLoad", detectors_module.CUSTOM_DETECTOR_REGISTRY)

    def test_env_var_used_when_no_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "env_plugin.py"), "w") as f:
                f.write(
                    "from dunetrace.detectors import BaseDetector\n"
                    "class EnvPlugin(BaseDetector):\n"
                    "    name = 'ENV_PLUGIN'\n"
                )
            with mock.patch.dict(os.environ, {"DUNETRACE_CUSTOM_DETECTORS_PATH": tmp}):
                loaded = load_custom_detector_plugins()
        self.assertEqual(loaded, 1)

    def test_two_files_defining_different_plugins_both_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "plugin_a.py"), "w") as f:
                f.write(
                    "from dunetrace.detectors import BaseDetector\n"
                    "class PluginA(BaseDetector):\n"
                    "    name = 'PLUGIN_A'\n"
                )
            with open(os.path.join(tmp, "plugin_b.py"), "w") as f:
                f.write(
                    "from dunetrace.detectors import BaseDetector\n"
                    "class PluginB(BaseDetector):\n"
                    "    name = 'PLUGIN_B'\n"
                )
            loaded = load_custom_detector_plugins(tmp)
        self.assertEqual(loaded, 2)
        self.assertIn("PluginA", detectors_module.CUSTOM_DETECTOR_REGISTRY)
        self.assertIn("PluginB", detectors_module.CUSTOM_DETECTOR_REGISTRY)


class TestGetDetectorsMergesPlugins(_RegistryIsolatedTestCase):
    def test_registered_plugin_appears_in_default_category(self):
        class _FakePlugin(BaseDetector):
            name = "FAKE_PLUGIN"

        detectors_module.CUSTOM_DETECTOR_REGISTRY["_FakePlugin"] = _FakePlugin
        result = detector_svc_detectors.get_detectors("default")
        names = [d.name for d in result]
        self.assertIn("FAKE_PLUGIN", names)

    def test_registered_plugin_appears_in_a_named_category_too(self):
        class _FakePlugin(BaseDetector):
            name = "FAKE_PLUGIN"

        detectors_module.CUSTOM_DETECTOR_REGISTRY["_FakePlugin"] = _FakePlugin
        result = detector_svc_detectors.get_detectors("web-research")
        names = [d.name for d in result]
        self.assertIn("FAKE_PLUGIN", names)

    def test_no_plugins_registered_returns_only_builtins(self):
        result = detector_svc_detectors.get_detectors("default")
        names = [d.name for d in result]
        self.assertNotIn("FAKE_PLUGIN", names)
        self.assertGreater(len(names), 0)  # built-ins are still there

    def test_plugin_instantiation_failure_is_skipped_not_raised(self):
        class _BrokenPlugin(BaseDetector):
            name = "BROKEN_PLUGIN"

            def __init__(self, **overrides):
                raise RuntimeError("boom at construction")

        detectors_module.CUSTOM_DETECTOR_REGISTRY["_BrokenPlugin"] = _BrokenPlugin
        result = detector_svc_detectors.get_detectors("default")  # must not raise
        names = [d.name for d in result]
        self.assertNotIn("BROKEN_PLUGIN", names)
        self.assertGreater(len(names), 0)  # built-ins still returned


if __name__ == "__main__":
    unittest.main(verbosity=2)
