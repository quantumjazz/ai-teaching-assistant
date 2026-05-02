class AssistantError(Exception):
    code = "assistant_error"
    status_code = 500


class UserInputError(AssistantError):
    code = "invalid_request"
    status_code = 400


class ConfigurationError(AssistantError):
    code = "configuration_error"
    status_code = 500


class KnowledgeBaseNotReady(AssistantError):
    code = "knowledge_base_not_ready"
    status_code = 503
