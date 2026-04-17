"""WSGI entry for production hosts (e.g. Render). Run from repo root; keeps cwd as project root."""
from __future__ import annotations

from SRC.app import server as application
