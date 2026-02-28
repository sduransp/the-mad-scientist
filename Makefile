PYTHON ?= python3
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PY := $(VENV_DIR)/bin/python

.PHONY: install smoke run

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -U pip setuptools wheel
	@if [ -f pyproject.toml ]; then \
		$(PIP) install -e .; \
	else \
		$(PIP) install -r requirements.txt; \
	fi

smoke:
	@. $(VENV_DIR)/bin/activate; \
	$(PY) - <<'PY'
import sys, importlib
print('Python:', sys.version)
pkgs = [
    'langchain', 'openai', 'pydantic', 'yaml', 'faiss', 'sklearn',
    'umap', 'hdbscan', 'bertopic', 'sentence_transformers', 'pypdf', 'pandas'
]
for p in pkgs:
    try:
        importlib.import_module(p)
        print('OK import:', p)
    except Exception as e:
        print('WARN import failed:', p, '-', e.__class__.__name__, str(e)[:200])
PY

run:
	@set -a; if [ -f .env ]; then . .env; fi; set +a; \
	. $(VENV_DIR)/bin/activate; \
	$(PY) - <<'PY'
import os
print('Mad Scientist runtime')
print('PROVIDER=', os.getenv('PROVIDER'))
print('CHAT_MODEL=', os.getenv('CHAT_MODEL'))
print('EMBEDDING_MODEL=', os.getenv('EMBEDDING_MODEL'))
PY
