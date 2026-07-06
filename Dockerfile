FROM python:3.12-slim

WORKDIR /code

# System deps needed by pypdf/chromadb build chain kept minimal on purpose.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p data/uploads data/chroma

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
