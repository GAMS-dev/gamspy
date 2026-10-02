"""
detect-secrets plugin that scans for GAMS access codes. It is registered in
.secrets.baseline.
"""

from __future__ import annotations

import re

from detect_secrets.filters import heuristic
from detect_secrets.plugins.base import BasePlugin, RegexBasedDetector


class GAMSAccessCodeDetector(RegexBasedDetector):
    """Scans for GAMS access codes (8-4-4-4-12 characters)."""

    secret_type = "GAMS Access Code"  # pragma: allowlist secret

    denylist = (
        re.compile(
            r"(?<![0-9A-Za-z])[0-9A-Za-z]{8}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}"
            r"-[0-9A-Za-z]{4}-[0-9A-Za-z]{12}(?![0-9A-Za-z])"
        ),
    )


def is_potential_uuid(secret: str, plugin: BasePlugin | None = None) -> bool:
    """
    Replaces detect_secrets.filters.heuristic.is_potential_uuid, which would
    discard every access code since access codes look like UUIDs.
    """
    # detect-secrets imports this file separately for the plugin and for the
    # filter, so isinstance does not work here.
    if getattr(plugin, "secret_type", None) == GAMSAccessCodeDetector.secret_type:
        return False

    return heuristic.is_potential_uuid(secret)
