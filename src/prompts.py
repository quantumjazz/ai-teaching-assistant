def uses_cyrillic(text):
    return any("\u0400" <= char <= "\u04ff" for char in text)


def no_answer_message(user_text):
    if uses_cyrillic(user_text):
        return (
            "Не знам. Не намерих достатъчно релевантна информация в "
            "индексираните учебни материали, за да отговоря уверено."
        )
    return (
        "I don't know. I could not find enough relevant information in "
        "the indexed course materials to answer that confidently."
    )


def answer_check_disabled_message(user_text=""):
    if uses_cyrillic(user_text):
        return (
            "Режимът за проверка на отговор изисква предишен учебен въпрос "
            "в този чат. Първо задайте въпрос по курса, след това изпратете "
            "отговора си с префикс a:."
        )
    return (
        "Answer-check mode needs a previous course question in this chat. Ask a "
        "course question first, then submit your answer with the a: prefix."
    )


def answer_check_missing_answer_message(user_text=""):
    if uses_cyrillic(user_text):
        return "Моля, добавете отговор след префикса a:."
    return "Please include an answer after the a: prefix."


def language_policy_instruction():
    return (
        "Match the language of the student's latest question. If the student "
        "uses Bulgarian, answer in Bulgarian. If the student uses English, "
        "answer in English."
    )


def build_messages(question_type, original_question, final_query, context, settings):
    instruction_prefix = settings.instructions.strip()
    if instruction_prefix:
        instruction_prefix += "\n\n"

    language_policy = language_policy_instruction()
    if question_type == "multiple_choice":
        prompt_instructions = (
            instruction_prefix
            + f"You are a precise TA in {settings.classname}. Construct a "
            + f"challenging multiple-choice question on {original_question} "
            + "using only the context. Present options A-D, then include "
            + f"Answer: and Explanation:. {language_policy}"
        )
    else:
        prompt_instructions = (
            instruction_prefix
            + f"You are {settings.assistant_name}, a TA for {settings.classname} "
            + f"({settings.classdescription}). Teach the answer using only the "
            + "provided context. Start with a direct definition in one or two "
            + "sentences, then explain the intuition behind the idea, and then "
            + "add a small example or implication when the context supports it. "
            + "Use short paragraphs or bullets instead of one dense block. "
            + "Explain technical terms in plain language. If the student's "
            + "wording is imprecise but the context contains a clearly related "
            + "term, briefly clarify the term before answering. Do not mention "
            + "authors or source titles unless they are needed to answer the "
            + 'question. If the answer is not found in context, say "I don\'t '
            + f'know" in the student\'s language. {language_policy}'
        )

    system_content = prompt_instructions
    if context:
        system_content += "\n\nContext:\n" + context

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": final_query},
    ]
