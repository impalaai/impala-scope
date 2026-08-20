"""Provider-neutral inference analytics without payload retention."""

from impala_scope.capture import configure, current_session_id, record_session, set_session_id
from impala_scope.hooks import install, uninstall

__all__ = ["configure", "current_session_id", "install", "record_session", "set_session_id", "uninstall"]

# Enable profiling for httpx-based clients on import.
install()
