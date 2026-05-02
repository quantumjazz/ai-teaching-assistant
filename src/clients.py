import os

from .env_utils import load_dotenv_if_available
from .errors import ConfigurationError


_configured_openai_module = None
_configured_openai_key = None
_dotenv_loaded = False


def load_openai_module(openai_module=None):
    if openai_module is not None:
        return openai_module
    try:
        import openai
    except ImportError as exc:
        raise ConfigurationError(
            "The openai package is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return openai


def load_faiss_module(faiss_module=None):
    if faiss_module is not None:
        return faiss_module
    try:
        import faiss
    except ImportError as exc:
        raise ConfigurationError(
            "The faiss-cpu package is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return faiss


def load_numpy_module(np_module=None):
    if np_module is not None:
        return np_module
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "The numpy package is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return np


def configure_openai(openai_module=None):
    global _configured_openai_key
    global _configured_openai_module
    global _dotenv_loaded

    if openai_module is not None:
        if not _dotenv_loaded:
            load_dotenv_if_available()
            _dotenv_loaded = True
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not configured. Add it to your environment or .env file."
            )
        openai_module.api_key = api_key
        return openai_module

    if _configured_openai_module is not None:
        return _configured_openai_module

    openai_module = load_openai_module(openai_module)
    if not _dotenv_loaded:
        load_dotenv_if_available()
        _dotenv_loaded = True
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not configured. Add it to your environment or .env file."
        )
    openai_module.api_key = api_key
    _configured_openai_module = openai_module
    _configured_openai_key = api_key
    return openai_module


def clear_client_cache():
    global _configured_openai_key
    global _configured_openai_module
    global _dotenv_loaded

    _configured_openai_module = None
    _configured_openai_key = None
    _dotenv_loaded = False
