import sys

from .errors import (
    AssistantError,
    ConfigurationError,
    KnowledgeBaseNotReady,
    UserInputError,
)
from .retrieval import load_faiss_resources
# These imports are intentionally re-exported for legacy callers and tests that
# still import the public service surface from src.main.
from .service import (
    ANSWER_CHECK_DISABLED_MESSAGE,
    AssistantResponse,
    answer_query,
    answer_query_with_sources,
)
from .settings import (
    DATA_DIR,
    SETTINGS_PATH,
    CourseSettings,
    load_course_settings,
    read_settings,
)


def main():
    user_input = input("Enter your prompt: ").strip()
    try:
        result = answer_query_with_sources(user_input)
    except AssistantError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print("\nFinal Answer:\n", result.response)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source.filename}, chunk {source.chunk_index}")


if __name__ == "__main__":
    main()
