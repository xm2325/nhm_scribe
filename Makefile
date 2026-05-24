.PHONY: install test demo clean

install:
	python -m pip install -e .

test:
	pytest

demo:
	python scripts/run_pipeline.py --config configs/demo_10.yaml

clean:
	rm -rf data/interim/* data/processed/* reports/*.md .pytest_cache
	mkdir -p data/interim/crops data/interim/ocr data/interim/llm data/processed
	touch data/interim/.gitkeep data/interim/crops/.gitkeep data/interim/ocr/.gitkeep data/interim/llm/.gitkeep data/processed/.gitkeep
