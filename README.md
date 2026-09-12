# FastAPI LID-176 Language Identification Service

A small FastAPI REST service and Python command-line interface for automatic language identification in JSON documents. It uses Facebook/Meta's multilingual `lid.176` model through `fastText`.

The service accepts an array of documents, including deeply nested objects and arrays. By default, it returns one dominant language for each document.

## Quick start

The following commands create a lightweight Conda environment, install the Python dependencies, download the model and start the API. This path usually resolves faster than creating the environment from `environment.yml`:

```bash
conda create -n lid176 python=3.12 pip -y
conda activate lid176
python -m pip install -r requirements.txt
mkdir -p models
wget -O models/lid.176.ftz \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
python main.py
```

If the `lid176` environment and the model are already available, use:

```bash
conda activate lid176
python -m pip install -r requirements.txt
python main.py
```

Send a minimal request from another terminal:

```bash
curl -X POST http://localhost:9292/detect \
  -H 'Content-Type: application/json' \
  -d '[{"text":"Buongiorno, questo e un testo italiano."}]'
```

Typical response:

```json
[
  {"language": "it", "probability": 0.99}
]
```

Interactive documentation: <http://localhost:9292/docs>

## Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [LID-176 model](#lid-176-model)
- [Running the service](#running-the-service)
- [REST API](#rest-api)
- [Response modes](#response-modes)
- [Command-line interface](#command-line-interface)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Technical notes](#technical-notes)
- [Troubleshooting](#troubleshooting)

## Features

- Language identification using the `lid.176` fastText model.
- Support for the language labels provided by the model, up to 176 labels.
- Input made of an array of JSON documents.
- Recursive processing of nested objects and arrays.
- `dominant` mode by default: one language per document.
- Optional `fields` mode: independent predictions for every text field.
- `top_k=1` by default; configurable up to `176`.
- Request and configuration validation with Pydantic.
- Fire-based CLI, Loguru logging and Rich terminal output.
- Built-in Swagger UI, ReDoc and OpenAPI schema.
- Direct startup with `python main.py` on port `9292`.

## Prerequisites

Recommended environment:

- Linux or WSL2;
- Python `3.12`;
- Conda, if you want to use the included environment definition;
- Internet access to download dependencies and the model.

Python 3.12 is recommended because some `fasttext-wheel` releases may not provide a ready-made build for newer Python versions.

## Installation

### Recommended: Conda

From the project directory:

```bash
conda env create -f environment.yml
conda activate lid176
```

If the environment already exists:

```bash
conda activate lid176
python -m pip install -r requirements.txt
```

### Alternative: virtual environment

With Python 3.12 available on the system:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Main dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `fasttext-wheel` | Python binding for fastText |
| `numpy<2.0` | Compatibility with `fasttext-wheel==0.9.2` |
| `pydantic` | Data and settings validation |
| `fire` | Command-line interface |
| `loguru` | Application logging |
| `rich` | Terminal output |

The complete version constraints are defined in `requirements.txt`.

## LID-176 model

By default, the service looks for the model at:

```text
models/lid.176.ftz
```

Download the compressed model with:

```bash
mkdir -p models
wget -O models/lid.176.ftz \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

To use a different path:

```bash
export LID_MODEL_PATH=/absolute/path/to/lid.176.ftz
```

The model is loaded lazily on the first request that needs inference and then reused by the process.

## Running the service

### Default startup

```bash
conda activate lid176
python main.py
```

The server listens on `0.0.0.0:9292`.

### Explicit CLI startup

```bash
python main.py serve
```

Start on a different port:

```bash
python main.py serve --port=8080
```

Show available Fire commands:

```bash
python main.py --help
```

## REST API

### `GET /health`

Returns a lightweight liveness response:

```bash
curl http://localhost:9292/health
```

Response:

```json
{"status": "ok"}
```

### `POST /detect`

The endpoint accepts either a plain JSON array or an object with a `documents` field.

#### Minimal input

```json
[
  {"text": "Buongiorno, questo e un testo italiano."}
]
```

#### Nested input

```json
[
  {
    "id": 42,
    "title": "Ciao mondo",
    "content": {
      "summary": "Questo testo descrive un documento italiano.",
      "paragraphs": [
        "La prima frase e italiana.",
        "Anche questa frase e italiana."
      ]
    },
    "metadata": {"source": "demo"}
  }
]
```

The endpoint recursively visits all non-empty strings inside objects and arrays.

#### Extended input with options

```json
{
  "documents": [
    {"title": "Ciao mondo", "content": "Questo e un testo italiano."}
  ],
  "top_k": 1,
  "result_mode": "dominant"
}
```

When the extended request format is used, the `documents` wrapper is preserved in the response.

## Response modes

### `dominant` (default)

This is the default mode. It returns one object for each document:

```json
{
  "language": "it",
  "probability": 0.98
}
```

The aggregation algorithm works as follows:

1. collect every non-empty string, including nested strings;
2. normalize whitespace and calculate each field's character length;
3. request up to `top_k` predictions for every field;
4. for each language, sum `probability * field_length`;
5. divide the sum by the total length of all non-empty text fields;
6. return the language with the highest weighted average.

Text length is therefore the field weight: a short tag has less influence than a long description. If a language does not occur in a field's top-k predictions, it contributes zero for that field.

If the document has an `id` key directly at its top level, it is treated as metadata rather than text. The identifier is preserved in the dominant response:

```json
[
  {"id": 42, "language": "it", "probability": 0.97}
]
```

A document without non-empty string fields produces:

```json
[
  {"language": null, "probability": 0.0}
]
```

The default `top_k` is `1`. It can be increased when several model predictions should contribute to the aggregation:

```json
{
  "documents": [{"text": "A short multilingual text"}],
  "top_k": 3,
  "result_mode": "dominant"
}
```

### `fields`: field-level details

Use `result_mode: "fields"` when you need one independent prediction array per field while preserving the original JSON structure.

Request:

```json
{
  "documents": [
    {
      "title": "Hello world",
      "body": {"text": "Questo e un testo italiano."}
    }
  ],
  "top_k": 2,
  "result_mode": "fields"
}
```

Response:

```json
{
  "documents": [
    {
      "title": [
        {"language": "en", "probability": 0.98},
        {"language": "de", "probability": 0.01}
      ],
      "body": {
        "text": [
          {"language": "it", "probability": 0.99},
          {"language": "la", "probability": 0.01}
        ]
      }
    }
  ]
}
```

In this mode:

- dictionary keys, object nesting and array positions are preserved;
- every string becomes an array of `{language, probability}` objects;
- empty strings and non-string values become `[]`;
- `top_k=1` remains the default unless explicitly overridden.

## Swagger UI and OpenAPI

With the server running:

- Swagger UI: <http://localhost:9292/docs>
- ReDoc: <http://localhost:9292/redoc>
- OpenAPI schema: <http://localhost:9292/openapi.json>

In Swagger UI:

1. open `POST /detect`;
2. click `Try it out`;
3. enter either a JSON array or the extended `documents` object;
4. click `Execute`;
5. inspect the result in `Response body`.

The endpoint includes examples for both dominant and field-level responses.

## Command-line interface

The CLI is implemented with Fire. JSON results are printed to stdout; Loguru messages are written to stderr.

### Single text

```bash
python main.py detect_text "Questo e un testo italiano"
python main.py detect_text "Questo e un testo italiano" --top_k=2
```

`detect_text` always returns the prediction list for the supplied text; `result_mode` does not apply to this command.

### JSON file

Create `input.json`:

```json
[
  {"title": "Hello world", "description": "This is an English description."},
  {"title": "Ciao mondo", "description": "Questa e una descrizione italiana."}
]
```

Process it in dominant mode:

```bash
python main.py detect_json input.json
```

Process it in detailed mode with two predictions per field:

```bash
python main.py detect_json input.json --top_k=2 --result_mode=fields
```

### Server options

```bash
python main.py serve \
  --host=0.0.0.0 \
  --port=9292 \
  --top_k=1 \
  --result_mode=dominant \
  --model_path=models/lid.176.ftz \
  --log_level=INFO
```

## Configuration

The following environment variables are supported:

| Variable | Default | Description |
|---|---|---|
| `LID_MODEL_PATH` | `models/lid.176.ftz` | Path to the fastText model |
| `HOST` | `0.0.0.0` | HTTP listening interface |
| `PORT` | `9292` | HTTP port |
| `TOP_K` | `1` | Predictions requested per field |
| `RESULT_MODE` | `dominant` | `dominant` or `fields` |
| `LOG_LEVEL` | `INFO` | Loguru logging level |

Example:

```bash
export LID_MODEL_PATH=/opt/models/lid.176.ftz
export PORT=8080
export TOP_K=1
export RESULT_MODE=dominant
python main.py
```

For an extended request, `top_k` and `result_mode` can be specified directly in the request body. If omitted, the Pydantic request defaults are used (`1` and `dominant`). For a plain JSON array, the environment defaults are used.

## Project structure

```text
149-lid/
├── main.py                 # FastAPI application and Fire CLI
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment for Python 3.12
├── models/
│   └── lid.176.ftz         # Compressed fastText model
└── README.md               # This documentation
```

## Technical notes

- The model is loaded once per process, on the first inference request.
- The `.ftz` file is the compressed inference model; local training is not required.
- Repeated whitespace and line breaks are normalized before inference.
- Probabilities are rounded to six decimal places in the JSON output.
- The service does not persist submitted documents.
- For production deployments, consider authentication, rate limiting, timeouts and centralized logging according to your environment.

## Troubleshooting

### Model not found

Verify that the file exists:

```bash
ls -lh models/lid.176.ftz
```

Alternatively, set `LID_MODEL_PATH` to an absolute path.

### `fasttext-wheel` installation error

Verify that Python 3.12 is active and reinstall the project dependencies:

```bash
python --version
python -m pip install -r requirements.txt
```

Keep the `numpy<2.0` constraint already present in `requirements.txt`.

### Port already in use

Start the service on a different port:

```bash
python main.py serve --port=8080
```

### HTTP 422 response

The body must be a JSON array or an object whose `documents` field is an array. `top_k` must be between `1` and `176`; `result_mode` must be `dominant` or `fields`.

### The server starts but model inference fails

Inspect the terminal logs and verify that the correct environment is active:

```bash
conda activate lid176
python -c "import fastapi, fasttext, numpy; print('environment OK')"
```

## License and model terms

This project uses `fastText` and the `lid.176` language model distributed by Facebook/Meta. Review the model and dependency licenses and terms of use before deploying the service in production.
