import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.env_utils import is_configured_env_value, load_dotenv_if_available
from src.index_status import get_index_status, supported_documents
from src.settings import PROJECT_ROOT, runtime_paths


INDEX_COMMANDS = (
    ("prepare documents", ("scripts/prepare_documents.py",)),
    ("embed documents", ("scripts/embed_documents.py",)),
    ("create FAISS index", ("scripts/create_final_data.py",)),
)


def bootstrap_index(project_root=None, env=None, runner=subprocess.run):
    explicit_project_root = project_root is not None
    paths = runtime_paths(project_root, env=env)
    if env is None:
        load_dotenv_if_available(paths.env_path, override=True)
        env = os.environ
        paths = runtime_paths(project_root, env=env)
        load_dotenv_if_available(paths.env_path, override=True)
        env = os.environ

    if not _env_flag(env, "AUTO_INDEX_ON_STARTUP", True):
        print("AUTO_INDEX_ON_STARTUP is disabled; skipping index bootstrap.")
        return "disabled"

    paths = runtime_paths(project_root, env=env)
    status = get_index_status(paths.course_root, env=env)
    if status.status == "ready":
        print("Knowledge base index is ready; skipping index bootstrap.")
        return "ready"

    if not supported_documents(status.documents):
        print("No supported course documents found; app will start with setup diagnostics.")
        return "skipped"

    if not is_configured_env_value(env.get("OPENAI_API_KEY")):
        print("OPENAI_API_KEY is not configured; app will start without building the index.")
        return "skipped"

    print(f"Knowledge base status is {status.status}; building index artifacts.")
    command_cwd = paths.course_root if explicit_project_root else PROJECT_ROOT
    for label, command in INDEX_COMMANDS:
        args = [sys.executable, *command]
        print(f"Running {label}: {' '.join(args)}")
        runner(args, cwd=str(command_cwd), check=True)
    return "built"


def _env_flag(env, name, default):
    value = env.get(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def main():
    try:
        bootstrap_index()
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
