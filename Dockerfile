FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY model.pkl .
COPY tfidf_vectorizer.pkl .

RUN pip install -r requirements.txt

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
