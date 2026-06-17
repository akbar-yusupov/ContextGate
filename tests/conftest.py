from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CONTEXTGATE_EMBEDDING_BACKEND", "deterministic")
os.environ.setdefault("CONTEXTGATE_DENSE_DIMENSION", "64")
os.environ.setdefault("CONTEXTGATE_LATE_DIMENSION", "32")
os.environ.setdefault("CONTEXTGATE_DATABASE_URL", "sqlite:///./.contextgate/test.db")
os.environ.setdefault("CONTEXTGATE_QDRANT_LOCAL_PATH", "./.contextgate/test-qdrant")
os.environ.setdefault("CONTEXTGATE_MLFLOW_TRACKING_URI", "./.contextgate/test-mlruns")
os.environ.setdefault("CONTEXTGATE_AUTH_ENABLED", "false")

Path(".contextgate").mkdir(exist_ok=True)
