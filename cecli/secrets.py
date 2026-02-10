"""Keyring-first secret management for cecli.

Secrets are retrieved from system keyring first, with environment variable fallback.
Keyring location: cecli/{env_var.lower()}
"""

import os
import re
from typing import Optional

KEYRING_AVAILABLE = False
try:
    import keyring
    from keyring.errors import KeyringError
    KEYRING_AVAILABLE = True
except ImportError:
    pass

DEFAULT_KEYRING_SERVICE = "cecli"


def get_secret(env_var: str, service: Optional[str] = None) -> Optional[str]:
    """Get secret from keyring or environment.

    Keyring location: service/username = cecli/{env_var.lower()}

    Args:
        env_var: Environment variable name (e.g., "OPENAI_API_KEY")
        service: Keyring service name (default: "cecli")

    Returns:
        Secret value or None
    """
    svc = service or DEFAULT_KEYRING_SERVICE
    username = env_var.lower()

    # Priority 1: Check keyring
    if KEYRING_AVAILABLE:
        try:
            secret = keyring.get_password(svc, username)
            if secret:
                return secret
        except KeyringError:
            pass

    # Priority 2: Environment fallback
    return os.environ.get(env_var)


def save_secret(env_var: str, secret: str, service: Optional[str] = None) -> bool:
    """Save secret to keyring."""
    if not KEYRING_AVAILABLE:
        raise RuntimeError("keyring module not installed")

    svc = service or DEFAULT_KEYRING_SERVICE
    username = env_var.lower()

    try:
        keyring.set_password(svc, username, secret)
        return True
    except KeyringError:
        return False


def delete_secret(env_var: str, service: Optional[str] = None) -> bool:
    """Delete secret from keyring."""
    if not KEYRING_AVAILABLE:
        return False

    svc = service or DEFAULT_KEYRING_SERVICE
    username = env_var.lower()

    try:
        keyring.delete_password(svc, username)
        return True
    except KeyringError:
        return False


def has_keyring() -> bool:
    """Check if keyring is available."""
    return KEYRING_AVAILABLE


def get_keyring_backend() -> Optional[str]:
    """Get the active keyring backend name."""
    if not KEYRING_AVAILABLE:
        return None
    return str(keyring.get_keyring())
