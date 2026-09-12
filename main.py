"""LID-176 language identification service.

This module exposes two interfaces over the same inference pipeline:

* a FastAPI application with ``/health`` and ``/detect`` endpoints;
* a Fire-powered command-line interface for serving the API and processing
  individual texts or JSON files.

The implementation deliberately keeps the data model permissive because the
service must accept arbitrary JSON documents. Validation is applied to the
request envelope and to configuration values, while the recursive processing
functions preserve the structure of each user document.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, TypeAlias

import fire
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

# A JSON document can contain any combination of objects, arrays and scalar
# values. ``Any`` is intentionally used at the boundary: rejecting a valid
# JSON scalar here would make the recursive transformer less useful.
JsonDocument: TypeAlias = Any
Prediction: TypeAlias = dict[str, float | str]
WeightedPredictions: TypeAlias = list[tuple[list[Prediction], int]]

DEFAULT_MODEL_PATH = Path("models/lid.176.ftz")
DEFAULT_TOP_K = 1
DEFAULT_RESULT_MODE = "dominant"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9292
DEFAULT_LOG_LEVEL = "INFO"

# Rich writes CLI results to stdout, while Loguru is configured to write logs
# to stderr. Keeping the streams separate makes the JSON output pipe-friendly.
console = Console()
_model: Any | None = None


class Settings(BaseModel):
    """Application settings shared by the REST server and the CLI.

    Attributes:
        model_path: Filesystem path of the compressed fastText model.
        host: Network interface on which Uvicorn should listen.
        port: TCP port on which Uvicorn should listen.
        top_k: Maximum number of model predictions requested per text field.
        result_mode: ``dominant`` aggregates each document; ``fields`` keeps
            the detailed, field-by-field representation.
        log_level: Loguru log level, such as ``INFO`` or ``DEBUG``.
    """

    model_path: Path = Field(
        DEFAULT_MODEL_PATH,
        description="Path to the lid.176.ftz model file.",
    )
    host: str = Field(
        DEFAULT_HOST,
        description="Network interface used by the HTTP server.",
    )
    port: int = Field(
        DEFAULT_PORT,
        ge=1,
        le=65535,
        description="TCP port used by the HTTP server.",
    )
    top_k: int = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=176,
        description="Number of language predictions requested per text field.",
    )
    result_mode: Literal["dominant", "fields"] = Field(
        DEFAULT_RESULT_MODE,
        description=(
            "'dominant' aggregates a document; 'fields' returns every field."
        ),
    )
    log_level: str = Field(
        DEFAULT_LOG_LEVEL,
        description="Loguru logging level.",
    )

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables.

        Returns:
            A validated :class:`Settings` instance. Missing variables use the
            application defaults defined above.

        Raises:
            ValueError: If an integer environment variable cannot be parsed.
            pydantic.ValidationError: If a parsed value violates a field
                constraint, for example a port outside the valid TCP range.
        """
        return cls(
            model_path=Path(os.getenv("LID_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            host=os.getenv("HOST", DEFAULT_HOST),
            port=int(os.getenv("PORT", str(DEFAULT_PORT))),
            top_k=int(os.getenv("TOP_K", str(DEFAULT_TOP_K))),
            result_mode=os.getenv("RESULT_MODE", DEFAULT_RESULT_MODE),
            log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        )


class DetectionRequest(BaseModel):
    """Extended request body accepted by the ``POST /detect`` endpoint.

    The endpoint also accepts a plain JSON array. This model is used when the
    caller needs to specify ``top_k`` or ``result_mode`` in the request body.

    Attributes:
        documents: Documents to classify. Objects and arrays may be nested.
        top_k: Number of predictions requested for every non-empty string.
        result_mode: Output representation selected for this request.
    """

    documents: list[JsonDocument] = Field(
        ...,
        description="Array of JSON documents, including nested structures.",
    )
    top_k: int = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=176,
        description="Number of languages requested per text field.",
    )
    result_mode: Literal["dominant", "fields"] = Field(
        DEFAULT_RESULT_MODE,
        description=(
            "'dominant' returns one language per document; "
            "'fields' returns field-level details."
        ),
    )


class LanguagePrediction(BaseModel):
    """Normalized prediction returned by the fastText model.

    Attributes:
        language: ISO-like language label produced by ``lid.176``.
        probability: Model confidence associated with the label.
    """

    language: str = Field(description="Language code, for example 'en'.")
    probability: float = Field(description="Probability assigned to the language.")


class DetectionResponse(BaseModel):
    """Response envelope used when the request uses the ``documents`` key.

    The service preserves this envelope so callers can distinguish the extended
    request format from the compact plain-array format.
    """

    documents: list[JsonDocument]


app = FastAPI(
    title="LID-176 Language Identification API",
    description="Language identification for nested JSON documents using fastText.",
    version="1.2.0",
)


class ModelNotFoundError(RuntimeError):
    """Raised when the configured fastText model file does not exist."""


class InvalidPayloadError(ValueError):
    """Raised when a request body is not one of the supported JSON shapes."""


class InvalidJsonFileError(ValueError):
    """Raised when a CLI input path is missing or contains invalid JSON."""


def configure_logging(log_level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure Loguru without contaminating machine-readable CLI output.

    Args:
        log_level: Case-insensitive Loguru level, for example ``INFO`` or
            ``DEBUG``.

    The default Loguru sink is removed so the application has one predictable
    stderr sink. JSON produced by Rich remains available on stdout for shell
    pipelines and scripts.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | {message}"
        ),
    )


def get_model(model_path: Path | None = None) -> Any:
    """Load and cache the fastText model on first use.

    Args:
        model_path: Optional explicit model path. If omitted, the
            ``LID_MODEL_PATH`` environment setting or the default path is used.

    Returns:
        The object returned by :func:`fasttext.load_model`.

    Raises:
        ModelNotFoundError: If the resolved model path does not exist.

    The import of ``fasttext`` is intentionally local. This keeps configuration
    and documentation commands lightweight and delays the native dependency
    loading until language detection is actually requested.
    """
    global _model

    # Loading the compressed model is relatively expensive. Reuse the same
    # instance for all requests handled by this process.
    if _model is not None:
        return _model

    resolved_model_path = model_path or Settings.from_env().model_path
    if not resolved_model_path.exists():
        raise ModelNotFoundError(
            f"Model not found at '{resolved_model_path}'. "
            "Set LID_MODEL_PATH to a valid .ftz file."
        )

    import fasttext

    logger.info("Loading LID-176 model from {}", resolved_model_path)
    _model = fasttext.load_model(str(resolved_model_path))
    return _model


def normalize_label(label: str) -> str:
    """Remove fastText's internal ``__label__`` prefix.

    Args:
        label: Raw label returned by fastText.

    Returns:
        The public language code without the implementation prefix.
    """
    return label.removeprefix("__label__")


def detect_language(
    text: str,
    top_k: int,
    model_path: Path | None = None,
) -> list[Prediction]:
    """Predict languages for one text value.

    Args:
        text: Input text. Whitespace is normalized before inference.
        top_k: Maximum number of predictions to return.
        model_path: Optional explicit model path passed to :func:`get_model`.

    Returns:
        A list of dictionaries containing ``language`` and ``probability``.
        Empty or whitespace-only input returns an empty list.
    """
    # Normalizing whitespace improves consistency for text copied from JSON
    # fields containing line breaks, indentation or repeated spaces.
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    labels, probabilities = get_model(model_path).predict(cleaned_text, k=top_k)
    return [
        LanguagePrediction(
            language=normalize_label(label),
            probability=round(float(probability), 6),
        ).model_dump()
        for label, probability in zip(labels, probabilities)
    ]


def is_excluded_field(field_name: str) -> bool:
    """Return whether a field name must bypass language identification.

    Args:
        field_name: JSON object key to inspect.

    Returns:
        ``True`` when the key is exactly ``id`` or contains ``date``, using a
        case-insensitive comparison; otherwise ``False``.

    The rule is applied at every nesting level. For example, both ``id`` and
    ``author_id`` are handled according to the exact-name rule only for the
    former, while ``creation_date`` and ``updatedDate`` match the date rule.
    """
    normalized_name = field_name.casefold()
    return normalized_name == "id" or "date" in normalized_name


def is_json_number(value: Any) -> bool:
    """Return whether a value is a JSON number rather than a boolean.

    Args:
        value: Value from a decoded JSON document.

    Returns:
        ``True`` for integers and floating-point values, including negative and
        fractional values. Booleans are explicitly excluded because Python's
        ``bool`` type is a subclass of ``int`` even though JSON treats booleans
        and numbers as different types.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def transform_leaf_values(
    value: Any,
    top_k: int,
    model_path: Path | None = None,
) -> Any:
    """Recursively produce the detailed field-level representation.

    Args:
        value: Current JSON node being visited.
        top_k: Maximum predictions requested for every string leaf.
        model_path: Optional explicit model path.

    Returns:
        A JSON-compatible value with the same dictionary keys and list layout.
        Fields named ``id`` or containing ``date`` are returned unchanged.
        Numeric values are also returned unchanged. Other string leaves become
        prediction arrays; empty strings and remaining non-string leaves become
        empty arrays.
    """
    if isinstance(value, dict):
        transformed: dict[str, Any] = {}
        for key, child in value.items():
            # Metadata fields are copied as a whole. This is important for a
            # date field whose value is a string: the date must never reach the
            # language model merely because it happens to be textual.
            if is_excluded_field(str(key)):
                transformed[key] = child
            else:
                transformed[key] = transform_leaf_values(child, top_k, model_path)
        return transformed

    if isinstance(value, list):
        return [transform_leaf_values(item, top_k, model_path) for item in value]

    if is_json_number(value):
        # Keep the original Python number so JSON serialization reproduces its
        # value instead of replacing it with an empty prediction array.
        return value

    if isinstance(value, str):
        return detect_language(value, top_k, model_path)

    # Booleans and null are valid JSON values, but are neither text nor the
    # explicitly protected numeric values covered above.
    return []


def collect_weighted_predictions(
    value: Any,
    top_k: int,
    model_path: Path | None = None,
) -> WeightedPredictions:
    """Collect model predictions and text weights from analyzable leaves.

    Args:
        value: Current JSON node.
        top_k: Maximum predictions requested per non-empty string.
        model_path: Optional explicit model path.

    Returns:
        A flat list of ``(predictions, weight)`` pairs, one pair for each
        non-empty string that is eligible for analysis. Metadata fields named
        ``id`` or containing ``date`` and every numeric value are excluded.
        Each weight is the number of characters after whitespace normalization.

    Flattening leaves makes the aggregation formula independent of nesting
    depth while leaving the original document untouched.
    """
    if isinstance(value, dict):
        collected: WeightedPredictions = []
        for key, child in value.items():
            # Excluded fields are skipped before recursion, so a nested date
            # object or an id string can never accidentally be analyzed.
            if is_excluded_field(str(key)):
                continue
            collected.extend(collect_weighted_predictions(child, top_k, model_path))
        return collected

    if isinstance(value, list):
        collected = []
        for child in value:
            collected.extend(collect_weighted_predictions(child, top_k, model_path))
        return collected

    if isinstance(value, str):
        cleaned_text = " ".join(value.split())
        if cleaned_text:
            # The normalized text determines both the model input and this
            # field's influence on the document-level weighted average.
            return [(detect_language(cleaned_text, top_k, model_path), len(cleaned_text))]

    # Numbers, booleans and null do not contain language-bearing text.
    return []


def detect_dominant_language(
    document: Any,
    top_k: int,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate the dominant language without analyzing protected fields.

    Args:
        document: Arbitrary JSON-compatible document.
        top_k: Maximum predictions considered for each eligible text field.
        model_path: Optional explicit model path.

    Returns:
        A dictionary with ``language`` and weighted ``probability``. If the
        document has a top-level ``id``, the identifier is included unchanged.
        Documents without eligible non-empty string fields return
        ``{"language": None, "probability": 0.0}``.

    The score is calculated as::

        score(language) = sum(probability * field_length) / total_text_length

    A language absent from a field's top-k predictions contributes zero for
    that field. Fields named ``id`` or containing ``date`` and all numeric
    values are excluded before this calculation begins.
    """
    has_document_id = isinstance(document, dict) and "id" in document
    document_id = document.get("id") if has_document_id else None
    predictions_with_weights = collect_weighted_predictions(document, top_k, model_path)

    if not predictions_with_weights:
        result: dict[str, Any] = {"language": None, "probability": 0.0}
        return {"id": document_id, **result} if has_document_id else result

    weighted_probability_sums: dict[str, float] = {}
    total_weight = 0

    # Each eligible field contributes probability multiplied by text length.
    # A language missing from a field's predictions therefore contributes zero.
    for field_predictions, weight in predictions_with_weights:
        total_weight += weight
        for prediction in field_predictions:
            language = str(prediction["language"])
            weighted_probability_sums[language] = weighted_probability_sums.get(
                language,
                0.0,
            ) + float(prediction["probability"]) * weight

    language, weighted_sum = max(
        weighted_probability_sums.items(),
        key=lambda item: item[1],
    )
    result = {
        "language": language,
        "probability": round(weighted_sum / total_weight, 6),
    }
    return {"id": document_id, **result} if has_document_id else result


def transform_dominant_document(
    document: Any,
    top_k: int,
    model_path: Path | None = None,
) -> Any:
    """Return a document-shaped result using its single dominant prediction.

    Args:
        document: Arbitrary JSON-compatible document.
        top_k: Maximum predictions considered while finding the dominant
            language.
        model_path: Optional explicit model path.

    Returns:
        The same object/list structure as ``document``. Protected fields
        (``id``, names containing ``date`` and numeric values) are copied
        exactly. Every eligible non-empty string is replaced with a one-item
        prediction array containing the document's dominant language and
        probability. Empty strings and unsupported scalar values become ``[]``.

    The dominant language is calculated once for the whole document and then
    reused for all eligible text leaves. This both matches the document-level
    contract and avoids running the model a second time for each field during
    reconstruction.
    """
    dominant = detect_dominant_language(document, top_k, model_path)
    if dominant["language"] is None:
        dominant_prediction: Prediction | None = None
    else:
        dominant_prediction = {
            "language": str(dominant["language"]),
            "probability": float(dominant["probability"]),
        }

    def rebuild(value: Any) -> Any:
        """Rebuild one node while applying the document-level result."""
        if isinstance(value, dict):
            rebuilt: dict[str, Any] = {}
            for key, child in value.items():
                if is_excluded_field(str(key)):
                    rebuilt[key] = child
                else:
                    rebuilt[key] = rebuild(child)
            return rebuilt

        if isinstance(value, list):
            return [rebuild(item) for item in value]

        if is_json_number(value):
            return value

        if isinstance(value, str):
            if not value.strip() or dominant_prediction is None:
                return []
            return [dominant_prediction.copy()]

        return []

    return rebuild(document)

def parse_detection_payload(payload: Any) -> tuple[DetectionRequest, bool]:
    """Normalize the two supported request shapes.

    Args:
        payload: Decoded JSON request body.

    Returns:
        A tuple containing the validated request and a boolean indicating
        whether the original body was a plain array. The boolean lets the
        endpoint preserve the caller's response envelope.

    Raises:
        InvalidPayloadError: If the body is neither an array nor a valid
            extended request object.
    """
    if isinstance(payload, list):
        settings = Settings.from_env()
        return (
            DetectionRequest(
                documents=payload,
                top_k=settings.top_k,
                result_mode=settings.result_mode,
            ),
            True,
        )

    if isinstance(payload, dict):
        try:
            return DetectionRequest.model_validate(payload), False
        except ValidationError as exc:
            raise InvalidPayloadError(str(exc)) from exc

    raise InvalidPayloadError(
        "Request body must be a JSON array or an object with a 'documents' field."
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response for monitoring systems."""
    return {"status": "ok"}


@app.post(
    "/detect",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "dominant_default": {
                            "summary": "Default dominant-language output",
                            "value": [
                                {
                                    "title": "Ciao mondo",
                                    "body": {"text": "Buongiorno a tutti"},
                                }
                            ],
                        },
                        "fields_top_k_2": {
                            "summary": "Detailed field-level output",
                            "value": {
                                "documents": [
                                    {"title": "Hello world", "body": "Ciao mondo"}
                                ],
                                "top_k": 2,
                                "result_mode": "fields",
                            },
                        },
                    }
                }
            }
        }
    },
)
def detect(payload: Any = Body(...)) -> Any:
    """Detect languages using the selected output mode.

    Args:
        payload: JSON array or extended request object received by FastAPI.

    Returns:
        A plain result array for plain-array input, or a ``documents`` wrapper
        for extended input. In ``dominant`` mode each result keeps the input
        document structure while replacing eligible text leaves with the
        document-level dominant prediction.

    Raises:
        HTTPException: ``422`` for invalid payloads and ``500`` when the model
            cannot be loaded.
    """
    try:
        request, input_was_plain_array = parse_detection_payload(payload)
        if request.result_mode == "fields":
            documents = [
                transform_leaf_values(document, request.top_k)
                for document in request.documents
            ]
        else:
            documents = [
                transform_dominant_document(document, request.top_k)
                for document in request.documents
            ]
    except InvalidPayloadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return (
        documents
        if input_was_plain_array
        else DetectionResponse(documents=documents).model_dump()
    )


class LidCli:
    """Fire command group exposing server and local inference commands."""

    def serve(
        self,
        host: str | None = None,
        port: int | None = None,
        model_path: str | None = None,
        top_k: int | None = None,
        result_mode: Literal["dominant", "fields"] | None = None,
        log_level: str | None = None,
    ) -> None:
        """Start the FastAPI application with optional CLI overrides.

        Args:
            host: Optional listening interface override.
            port: Optional TCP port override.
            model_path: Optional path to the ``.ftz`` model.
            top_k: Optional number of predictions per text field.
            result_mode: Optional default REST output mode.
            log_level: Optional Loguru level.
        """
        environment_settings = Settings.from_env()
        settings = environment_settings.model_copy(
            update={
                "host": host or environment_settings.host,
                "port": port or environment_settings.port,
                "model_path": Path(model_path)
                if model_path
                else environment_settings.model_path,
                "top_k": top_k or environment_settings.top_k,
                "result_mode": result_mode or environment_settings.result_mode,
                "log_level": log_level or environment_settings.log_level,
            }
        )
        configure_logging(settings.log_level)
        os.environ.update(
            {
                "LID_MODEL_PATH": str(settings.model_path),
                "PORT": str(settings.port),
                "TOP_K": str(settings.top_k),
                "RESULT_MODE": settings.result_mode,
            }
        )
        logger.info(
            "Starting server on {}:{} (top_k={}, result_mode={})",
            settings.host,
            settings.port,
            settings.top_k,
            settings.result_mode,
        )
        uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)

    def detect_text(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        model_path: str | None = None,
        log_level: str = DEFAULT_LOG_LEVEL,
    ) -> None:
        """Print language predictions for one text string as JSON.

        Args:
            text: Text to classify.
            top_k: Maximum number of language predictions.
            model_path: Optional path to the fastText model.
            log_level: Loguru level for this command.
        """
        configure_logging(log_level)
        console.print_json(
            data=detect_language(
                text,
                top_k,
                Path(model_path) if model_path else None,
            )
        )

    def detect_json(
        self,
        input_file: str,
        top_k: int | None = None,
        result_mode: Literal["dominant", "fields"] | None = None,
        model_path: str | None = None,
        log_level: str = DEFAULT_LOG_LEVEL,
    ) -> None:
        """Process a JSON file using the same modes as the REST endpoint.

        Args:
            input_file: Path to a plain-array or extended JSON request.
            top_k: Optional command-line override for predictions per field.
            result_mode: Optional ``dominant`` or ``fields`` override.
            model_path: Optional path to the fastText model.
            log_level: Loguru level for this command.

        Raises:
            InvalidJsonFileError: If the file is absent or contains malformed
                JSON.
        """
        configure_logging(log_level)
        path = Path(input_file)
        if not path.exists():
            raise InvalidJsonFileError(f"JSON file not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidJsonFileError(
                f"Invalid JSON in {path}: {exc}"
            ) from exc

        request, input_was_plain_array = parse_detection_payload(payload)
        effective_top_k = top_k or request.top_k
        effective_result_mode = result_mode or request.result_mode
        selected_model_path = Path(model_path) if model_path else None

        if effective_result_mode == "fields":
            documents = [
                transform_leaf_values(document, effective_top_k, selected_model_path)
                for document in request.documents
            ]
        else:
            documents = [
                transform_dominant_document(
                    document,
                    effective_top_k,
                    selected_model_path,
                )
                for document in request.documents
            ]

        console.print_json(
            data=documents if input_was_plain_array else {"documents": documents}
        )


def main() -> None:
    """Dispatch to the default server or to a Fire CLI command.

    Running ``python main.py`` without arguments starts the server. Supplying
    arguments delegates command parsing to Fire, for example
    ``python main.py detect_text "Hello world"``.
    """
    cli = LidCli()
    if len(sys.argv) == 1:
        cli.serve()
    else:
        fire.Fire(cli)


if __name__ == "__main__":
    main()
