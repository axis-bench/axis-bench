"""The renderer's smoke tests run in the renderer's environment (mink installed); elsewhere this
directory is not collected."""
from __future__ import annotations

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("mink") is not None else ["*"]
