import argparse
import json
import sys

from src.clients import configure_openai
from src.evaluation import load_eval_cases, score_retrieval, summarize_results
from src.retrieval import get_context_from_query, load_faiss_resources
from src.settings import DATA_DIR, SETTINGS_PATH, load_course_settings


def evaluate_cases(cases_path, settings_path=SETTINGS_PATH, data_dir=DATA_DIR):
    settings = load_course_settings(settings_path)
    index, metadata = load_faiss_resources(data_dir=data_dir)
    openai_module = configure_openai()

    results = []
    for case in load_eval_cases(cases_path):
        retrieved = get_context_from_query(
            case.question,
            index=index,
            metadata=metadata,
            settings=settings,
            openai_module=openai_module,
        )
        results.append(score_retrieval(case, retrieved))
    return summarize_results(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument(
        "--cases",
        default="eval/rag_cases.json",
        help="Path to a JSON file of RAG evaluation cases.",
    )
    args = parser.parse_args()

    try:
        summary = evaluate_cases(args.cases)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
