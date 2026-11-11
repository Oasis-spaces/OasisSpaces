#!/usr/bin/env python3
"""The Kaggle command line, for a session whose network proxy adds the Kaggle token.

In a cloud session the Kaggle API token is saved as a network secret: the proxy
puts it on every request to www.kaggle.com, and no copy exists in the session.
The stock CLI refuses to start without a token it can check itself, so this runs
it with a placeholder token and the check skipped; the proxy supplies the real
one. Elsewhere (a token in KAGGLE_API_TOKEN or ~/.kaggle) use `kaggle` directly.

    KAGGLE_USERNAME=<you> python3 tools/kaggle_cli.py kernels list --mine
"""
import os
import sys

os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_proxy_injected")
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

KaggleApi._introspect_token = lambda self, token: os.environ.get("KAGGLE_USERNAME", "unknown")

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
