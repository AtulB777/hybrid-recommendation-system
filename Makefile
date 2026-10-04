.PHONY: install data train evaluate serve test ui docker-up quickstart clean

install:
	pip install -r requirements-dev.txt

data:            ## generate synthetic CSVs
	python -m data.generate_synthetic

train:           ## OFFLINE: ingest, tune, evaluate, fit, precompute recommendations
	python -m training.train

evaluate:        ## compare all models on the held-out test split
	python -m training.evaluate --k 5 10 20

serve:           ## ONLINE: start the API on :8000
	uvicorn app.main:app --reload --port 8000

test:
	pytest

ui:              ## optional Streamlit demo (API must be running)
	streamlit run streamlit_app.py

docker-up:
	docker compose up --build

quickstart: install train   ## then run `make serve`
	@echo "Now run: make serve   (docs at http://localhost:8000/docs)"

clean:
	rm -f data/recsys.db artifacts/*.joblib
