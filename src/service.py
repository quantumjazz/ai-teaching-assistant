from dataclasses import dataclass, replace
from typing import List

from .clients import configure_openai
from .errors import UserInputError
from .llm import chat_completion, check_syllabus, verify_answer
from .prompts import (
    answer_check_disabled_message,
    answer_check_missing_answer_message,
    build_messages,
    language_policy_instruction,
    no_answer_message,
)
from .retrieval import Source, get_context_from_query, load_faiss_resources
from .settings import load_course_settings, runtime_paths


ANSWER_CHECK_DISABLED_MESSAGE = answer_check_disabled_message()


@dataclass(frozen=True)
class AssistantResponse:
    response: str
    sources: List[Source]

    def to_dict(self):
        return {
            "response": self.response,
            "sources": [source.to_dict() for source in self.sources],
        }


def answer_query(
    user_input,
    settings_path=None,
    data_dir=None,
    openai_module=None,
    faiss_module=None,
    verify=True,
    conversation_state=None,
):
    return answer_query_with_sources(
        user_input,
        settings_path=settings_path,
        data_dir=data_dir,
        openai_module=openai_module,
        faiss_module=faiss_module,
        verify=verify,
        conversation_state=conversation_state,
    ).response


def answer_query_with_sources(
    user_input,
    settings_path=None,
    data_dir=None,
    openai_module=None,
    faiss_module=None,
    verify=True,
    conversation_state=None,
):
    user_input = user_input.strip()
    if not user_input:
        raise UserInputError("No query provided")

    paths = runtime_paths()
    settings_path = settings_path or paths.settings_path
    data_dir = data_dir or paths.data_dir

    if user_input.lower().startswith("a:"):
        return _answer_check(
            user_input[2:].strip(),
            settings_path=settings_path,
            openai_module=openai_module,
            conversation_state=conversation_state,
        )

    question_type = "normal"
    if user_input.lower().startswith("m:"):
        question_type = "multiple_choice"
        user_input = user_input[2:].strip()
        if not user_input:
            raise UserInputError("No query provided")

    settings = load_course_settings(settings_path)
    index, metadata = load_faiss_resources(data_dir=data_dir, faiss_module=faiss_module)
    openai_module = configure_openai(openai_module)

    original_question = user_input
    if question_type == "normal" and check_syllabus(user_input, settings, openai_module):
        original_question = (
            f"I may be asking about the syllabus for {settings.classname}. {user_input}"
        )
    if question_type == "normal" and conversation_state and conversation_state.latest():
        original_question = (
            "Use this previous chat only if it helps resolve references in the "
            f"new question:\n{conversation_state.summary()}\n\n"
            f"New question: {original_question}"
        )

    retrieved = get_context_from_query(
        original_question,
        index=index,
        metadata=metadata,
        settings=settings,
        openai_module=openai_module,
    )
    if not retrieved.answerable:
        return AssistantResponse(
            no_answer_message(user_input),
            retrieved.sources,
        )

    if question_type == "multiple_choice":
        final_query = f"Construct a challenging multiple-choice question on: {original_question}"
    else:
        final_query = original_question

    messages = build_messages(
        question_type,
        original_question=original_question,
        final_query=final_query,
        context=retrieved.text,
        settings=settings,
    )
    reply = chat_completion(messages, settings=settings, openai_module=openai_module)
    sources = retrieved.sources

    if verify and question_type != "multiple_choice":
        verified = verify_answer(original_question, reply, settings, openai_module)
        if not verified:
            alt_retrieved = get_context_from_query(
                original_question,
                index=index,
                metadata=metadata,
                settings=_with_minimum_chunks(settings, 5),
                openai_module=openai_module,
            )
            if alt_retrieved.answerable:
                followup_messages = build_messages(
                    question_type,
                    original_question=original_question,
                    final_query=final_query,
                    context=alt_retrieved.text,
                    settings=settings,
                )
                followup_reply = chat_completion(
                    followup_messages,
                    settings=settings,
                    openai_module=openai_module,
                )
                if verify_answer(original_question, followup_reply, settings, openai_module):
                    reply = followup_reply
                    sources = alt_retrieved.sources

    response = AssistantResponse(reply, sources)
    if conversation_state and question_type != "multiple_choice":
        conversation_state.add_turn(
            user=user_input,
            assistant=reply,
            context=retrieved.text,
            sources=[source.to_dict() for source in sources],
        )
    return response


def _answer_check(user_answer, settings_path=None, openai_module=None, conversation_state=None):
    if not conversation_state or not conversation_state.latest():
        return AssistantResponse(answer_check_disabled_message(user_answer), [])
    if not user_answer:
        return AssistantResponse(answer_check_missing_answer_message(user_answer), [])

    settings_path = settings_path or runtime_paths().settings_path
    settings = load_course_settings(settings_path)
    openai_module = configure_openai(openai_module)
    previous = conversation_state.latest()
    messages = [
        {
            "role": "system",
            "content": (
                f"You are {settings.assistant_name}, a TA for {settings.classname}. "
                "Using only the provided context and prior assistant answer, tell "
                "the student whether their answer is correct. Give a concise "
                "rationale and mention any missing correction."
                f" {language_policy_instruction()}"
                "\n\nContext:\n"
                f"{previous.context}"
                "\n\nPrior assistant answer:\n"
                f"{previous.assistant}"
            ),
        },
        {
            "role": "user",
            "content": f"Student answer to check: {user_answer}",
        },
    ]
    reply = chat_completion(messages, settings=settings, openai_module=openai_module)
    return AssistantResponse(reply, [])


def _with_minimum_chunks(settings, minimum_chunks):
    return replace(settings, num_chunks=max(settings.num_chunks, minimum_chunks))
