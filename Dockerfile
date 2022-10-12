FROM python:3.7-slim AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    pip install -r requirements.txt

COPY setup.py MANIFEST.in pyproject.toml LICENSE.txt README.md .
COPY repsys repsys
RUN pip install .

FROM python:3.7-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

VOLUME /app
WORKDIR /app

EXPOSE 3001
ENTRYPOINT ["repsys"]
CMD ["server"]
