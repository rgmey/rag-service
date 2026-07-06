import os

# Config raises at import time if this is missing, so set a dummy value
# before any app module gets imported by the test suite. Also point
# storage paths at test-only locations so test runs never touch real data.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "data/test_chroma")
os.environ.setdefault("JOBS_DB_PATH", "data/test_jobs.db")
