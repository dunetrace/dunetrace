"""
Asserts dunetrace_schemas' enums stay value-for-value in sync with the SDK's
(dunetrace.models). The two packages have no import dependency on each other
by design — dunetrace-schemas must not pull pydantic into the SDK's zero-
dependency import chain, and ingest_svc (a dunetrace-schemas consumer)
deliberately does not depend on the SDK package at runtime. This test is the
enforcement mechanism instead: it fails CI the moment the two drift.

Requires packages/sdk-py on the path in addition to this package — see
PYTHONPATH in the Makefile / CLAUDE.md.

Run:
    PYTHONPATH=packages/sdk-py:packages/schemas-py \
        python -m pytest packages/schemas-py/tests/test_sdk_parity.py -v
"""

from __future__ import annotations

import unittest


class TestSdkParity(unittest.TestCase):
    def test_event_type_values_match_sdk(self):
        from dunetrace.models import EventType as SdkEventType
        from dunetrace_schemas.enums import EventType as SchemaEventType

        sdk_values = {e.value for e in SdkEventType}
        schema_values = {e.value for e in SchemaEventType}
        self.assertEqual(sdk_values, schema_values)

    def test_severity_values_match_sdk(self):
        from dunetrace.models import Severity as SdkSeverity
        from dunetrace_schemas.enums import Severity as SchemaSeverity

        sdk_values = {e.value for e in SdkSeverity}
        schema_values = {e.value for e in SchemaSeverity}
        self.assertEqual(sdk_values, schema_values)

    def test_failure_type_values_match_sdk(self):
        from dunetrace.models import FailureType as SdkFailureType
        from dunetrace_schemas.enums import FailureType as SchemaFailureType

        sdk_values = {e.value for e in SdkFailureType}
        schema_values = {e.value for e in SchemaFailureType}
        self.assertEqual(sdk_values, schema_values)

    def test_event_type_names_match_sdk(self):
        """Names matching too (not just values) — catches a renamed member with the same value."""
        from dunetrace.models import EventType as SdkEventType
        from dunetrace_schemas.enums import EventType as SchemaEventType

        sdk_names = {e.name for e in SdkEventType}
        schema_names = {e.name for e in SchemaEventType}
        self.assertEqual(sdk_names, schema_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
