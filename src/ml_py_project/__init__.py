# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""ml-py-project package."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("ml_py_project")
