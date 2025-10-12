build: normal_grain_merge/normal_grain_merge.c
	python3 -m build

install:
	pip install --force-reinstall .
	cp venv/lib/python3.12/site-packages/normal_grain_merge/*.so normal_grain_merge/

all: build install
