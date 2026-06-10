"""Financial Data Analysis — compatibility entry point.

This file exists so scripts referencing ``dashboard/financial_analyst_app.py``
continue to work.  The actual UI is in ``financial_analyst_dashboard.py``.
"""

from __future__ import annotations

import os
import sys

from dashboard import PARENT_DIR, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(str(PARENT_DIR))

# Redirect to the actual dashboard.
_dashboard_path = os.path.join(os.path.dirname(__file__), "financial_analyst_dashboard.py")
with open(_dashboard_path) as _f:
    exec(compile(_f.read(), _dashboard_path, "exec"))
