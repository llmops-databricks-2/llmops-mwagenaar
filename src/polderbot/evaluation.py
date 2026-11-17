"""Evaluate Polderbot responses against the evaluation dataset."""

import json
from pathlib import Path

import mlflow.genai
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers import Correctness, Guidelines, scorer
from mlflow.genai.scorers.base import Scorer

from polderbot.agent import PolderAgent

_EVALUATION_DATA_PATH = Path(__file__).parent / "resources" / "evaluation_data.json"
_WORD_COUNT_THRESHOLD = 20


def evaluate_agent(agent: PolderAgent, judge_endpoint: str | None) -> EvaluationResult:
    """Evaluate an existing agent with the configured Databricks judge.

    Args:
        agent: Agent used to generate an independent answer for each question.
        judge_endpoint: Databricks model endpoint used by the scorers.

    Returns:
        MLflow evaluation results, including per-question assessments.

    Raises:
        ValueError: If the judge endpoint has not been configured.
        OSError: If the evaluation dataset cannot be read.
        json.JSONDecodeError: If the evaluation dataset is not valid JSON.

    """
    if not judge_endpoint or not judge_endpoint.strip():
        raise ValueError("Configure judge_endpoint before running the evaluation.")

    return mlflow.genai.evaluate(
        data=_load_evaluation_data(),
        predict_fn=agent.generate_message,
        scorers=_get_scorers(judge_endpoint.strip()),
    )


@scorer
def word_count_check(outputs: str) -> bool:
    """Check whether the response contains more than ten space-separated words.

    Args:
        outputs: Agent response to evaluate.

    Returns:
        Whether the response exceeds the word-count threshold.

    """
    return len([word for word in outputs.split(" ") if word]) > _WORD_COUNT_THRESHOLD


def _get_scorers(judge_endpoint: str) -> list[Scorer]:
    """Create correctness, language and word-count scorers.

    Args:
        judge_endpoint: Databricks model endpoint used by the scorers.

    Returns:
        Scorers for expected facts, response language and word count.

    """
    model = f"databricks:/{judge_endpoint}"

    return [
        Correctness(model=model),
        Guidelines(
            name="language",
            guidelines="The response must be in the same language as the user's input message.",
            model=model,
        ),
        word_count_check,
    ]


def _load_evaluation_data() -> list[dict[str, dict[str, str]]]:
    """Load the bundled evaluation dataset.

    Returns:
        Questions and expected responses for MLflow evaluation.

    Raises:
        OSError: If the evaluation dataset cannot be read.
        json.JSONDecodeError: If the evaluation dataset is not valid JSON.

    """
    with _EVALUATION_DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)
