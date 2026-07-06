# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# sys_module/debug_logger.py
"""Debug logging utilities for HRCS standalone scripts (e.g. newTask.py)."""

import os
import sys
from datetime import datetime
from typing import Optional, TextIO


class _TeeStream:
    """Write to multiple streams (console + log file)."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


_log_file_handle: Optional[TextIO] = None
_original_stdout = None
_original_stderr = None
_debug_enabled = False


def is_debug_enabled() -> bool:
    return _debug_enabled


def setup_debug_logging(enable: bool, log_file: Optional[str] = None) -> Optional[str]:
    """
    Enable or disable debug logging.

    When enable=True, stdout/stderr are tee'd to log_file (and still shown on console).
    When enable=False, restore original streams if previously redirected.

    Returns:
        Absolute path to the log file when debug is enabled, else None.
    """
    global _log_file_handle, _original_stdout, _original_stderr, _debug_enabled

    teardown_debug_logging()

    _debug_enabled = enable
    if not enable:
        return None

    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"newTask_debug_{timestamp}.log")

    log_file = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    _log_file_handle = open(log_file, "w", encoding="utf-8", buffering=1)
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = _TeeStream(_original_stdout, _log_file_handle)
    sys.stderr = _TeeStream(_original_stderr, _log_file_handle)

    print(f"[DEBUG] Logging enabled — output tee'd to: {log_file}")
    return log_file


def teardown_debug_logging():
    """Restore original stdout/stderr and close the log file."""
    global _log_file_handle, _original_stdout, _original_stderr, _debug_enabled

    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr

    if _log_file_handle is not None:
        try:
            _log_file_handle.close()
        except Exception:
            pass

    _log_file_handle = None
    _original_stdout = None
    _original_stderr = None
    _debug_enabled = False
