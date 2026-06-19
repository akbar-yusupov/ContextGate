from __future__ import annotations

import os
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="contextgate-tests-")).resolve()

os.environ["CONTEXTGATE_ENV_FILE"] = ""
os.environ["CONTEXTGATE_ENVIRONMENT"] = "test"
os.environ["CONTEXTGATE_EMBEDDING_BACKEND"] = "deterministic"
os.environ["CONTEXTGATE_DENSE_DIMENSION"] = "64"
os.environ["CONTEXTGATE_LATE_DIMENSION"] = "32"
os.environ["CONTEXTGATE_DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["CONTEXTGATE_QDRANT_LOCAL_PATH"] = str(TEST_ROOT / "qdrant")
os.environ["CONTEXTGATE_MLFLOW_TRACKING_URI"] = str(TEST_ROOT / "mlruns")
os.environ["CONTEXTGATE_AUTH_ENABLED"] = "false"
os.environ["CONTEXTGATE_RATE_LIMIT_ENABLED"] = "false"
