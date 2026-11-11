#!/usr/bin/env python3
"""The Kaggle command line, for a session whose network proxy adds the Kaggle token.

In a cloud session the Kaggle API token is saved as a network secret: the proxy
puts it on every request to www.kaggle.com, and no copy exists in the session.
The stock CLI refuses to start without a token it can check itself, so this runs
it with a placeholder token and the check skipped; the proxy supplies the real
one, and the API is reached on www.kaggle.com, the host the token is saved
for. Elsewhere (a token in KAGGLE_API_TOKEN or ~/.kaggle) use `kaggle` directly.

    KAGGLE_USERNAME=<you> python3 tools/kaggle_cli.py kernels list --mine
"""
import os
import sys

os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_proxy_injected")
# The CLI checks the token as soon as the kaggle package is imported, so the check
# is answered here first: the token is taken as valid, for KAGGLE_USERNAME.
from kagglesdk.security.services.oauth_service import OAuthClient  # noqa: E402
from kagglesdk.security.types.oauth_service import IntrospectTokenResponse  # noqa: E402


def _introspect(self, request=None):
    response = IntrospectTokenResponse()
    response.active = True
    response.username = os.environ.get("KAGGLE_USERNAME", "")
    return response


if not os.path.exists(os.path.expanduser("~/.kaggle/access_token")):
    OAuthClient.introspect_token = _introspect
    # The token is saved for www.kaggle.com, which serves the same API under /api/v1.
    from kagglesdk import kaggle_env  # noqa: E402

    kaggle_env._env_to_endpoint[kaggle_env.KaggleEnv.PROD] = "https://www.kaggle.com/api"

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
