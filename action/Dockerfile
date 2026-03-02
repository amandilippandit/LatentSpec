FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /action

# Install latentspec from the same source tree the action ships with
COPY pyproject.toml /action/pyproject.toml
COPY README.md /action/README.md
COPY src/ /action/src/

RUN pip install --upgrade pip && pip install /action

COPY action/entrypoint.py /action/entrypoint.py
RUN chmod +x /action/entrypoint.py

ENTRYPOINT ["python", "/action/entrypoint.py"]
