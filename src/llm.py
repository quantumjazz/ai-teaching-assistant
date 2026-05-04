from .settings import CourseSettings


def chat_completion(messages, settings: CourseSettings, openai_module=None, **kwargs):
    response = openai_module.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        **_chat_completion_kwargs(settings.chat_model, kwargs),
    )
    return response.choices[0].message.content.strip()


def _chat_completion_kwargs(model, kwargs):
    kwargs = dict(kwargs)
    if _uses_reasoning_chat_parameters(model):
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if kwargs.get("temperature") == 0.0:
            kwargs.pop("temperature")
    return kwargs


def _uses_reasoning_chat_parameters(model):
    model = (model or "").lower()
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def verify_answer(original_question, answer, settings, openai_module=None):
    prompt = [
        {
            "role": "system",
            "content": "Just say 'Yes' or 'No'. Do not give any other answer.",
        },
        {
            "role": "user",
            "content": (
                f"User: {original_question}\nAssistant: {answer}\n"
                "Was the Assistant able to answer the user's question?"
            ),
        },
    ]
    verdict = chat_completion(
        prompt,
        settings=settings,
        openai_module=openai_module,
        max_tokens=5,
        temperature=0.0,
    ).lower()
    return verdict.startswith("y")


def check_syllabus(question, settings, openai_module=None):
    prompt = [
        {
            "role": "user",
            "content": (
                f"This question is from a student in {settings.classname} taught "
                f"by {settings.professor} with the help of {settings.assistants}. "
                f"The class is {settings.classdescription}. Is this question "
                f"likely about syllabus details? Answer Yes or No: {question}"
            ),
        }
    ]
    result = chat_completion(
        prompt,
        settings=settings,
        openai_module=openai_module,
        max_tokens=5,
        temperature=0.0,
    ).lower()
    return result.startswith("y")


def translate_retrieval_query(question, settings, openai_module=None):
    prompt = [
        {
            "role": "system",
            "content": (
                "Convert the student's course question into one concise English "
                "retrieval search query. Preserve technical terms and named "
                "concepts. Do not answer the question. Return only the query."
            ),
        },
        {"role": "user", "content": question},
    ]
    return chat_completion(
        prompt,
        settings=settings,
        openai_module=openai_module,
        max_tokens=80,
        temperature=0.0,
    )
