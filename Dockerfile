FROM python:3.12 as venv
WORKDIR /app

RUN apt update \
    && apt install curl --no-install-recommends -y \ 
    && curl -sSL https://install.python-poetry.org | python3 - 
    
ENV PATH /root/.local/bin:$PATH

COPY pyproject.toml poetry.lock ./
RUN python -m venv --copies /app/venv \
&& . /app/venv/bin/activate \
&& poetry install --only main



FROM python:3.12-slim as prod
WORKDIR /app
COPY --from=venv /app/venv /app/venv

RUN apt-get update \
    && apt-get install libopus0 --no-install-recommends -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /app/venv/bin:${PATH}
COPY ./src ./src

CMD python src/main.py
