"""Suite-wide defaults for the SDK tests.

Implicit runs (dunetrace.implicit) are on by default for users, but every
test written before they existed assumes the attach-only behaviour: a patched
call outside a run records nothing. With implicit runs on, those tests left
implicit runs open on clients they never shut down; each closed on its idle
timer ~30s later and shipped from that leaked client while an unrelated test
was counting HTTP calls. Default them off here; tests/test_implicit_runs.py
clears every DUNETRACE_* variable in setUp, so it runs with the product
default (on).
"""

import os

os.environ.setdefault("DUNETRACE_IMPLICIT_RUNS", "0")
