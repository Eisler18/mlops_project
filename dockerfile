FROM python:3.12-slim

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN pip install --no-cache-dir uv
RUN uv sync --no-dev

COPY . . 

EXPOSE 8000
ENTRYPOINT [ "uvicorn", "src/inference_api.py:app" ]
#CMD [ "uv", "src/main.py" ]
