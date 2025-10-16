# Simple cross-platform task runner (install: https://github.com/casey/just)
set shell := ["bash", "-cu"]

default:
    @just --list

venv:
    python -m venv .venv

install:
    . .venv/Scripts/activate || . .venv/bin/activate; pip install -U pip; pip install -e ".[dev]"

fmt:
    . .venv/Scripts/activate || . .venv/bin/activate; ruff format .; isort .; black .

lint:
    . .venv/Scripts/activate || . .venv/bin/activate; ruff check .

typecheck:
    . .venv/Scripts/activate || . .venv/bin/activate; mypy

test:
    . .venv/Scripts/activate || . .venv/bin/activate; pytest -q

gui:
    . .venv/Scripts/activate || . .venv/bin/activate; python -m pmc_app
