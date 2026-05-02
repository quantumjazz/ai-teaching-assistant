import os
import secrets
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, session

from .env_utils import env_flag, is_configured_env_value, load_dotenv_if_available
from .health import get_health_report
from .index_status import get_index_status
from .main import AssistantError, answer_query_with_sources
from .sessions import conversation_store

# Configure Flask to look for templates in the project root's "templates" folder.
template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
load_dotenv_if_available()
app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
secret_key = os.environ.get("FLASK_SECRET_KEY")
app.secret_key = secret_key if is_configured_env_value(secret_key) else secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    SESSION_COOKIE_SECURE=env_flag("SESSION_COOKIE_SECURE", False),
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setup')
def setup():
    report = get_health_report().to_dict()
    return render_template('setup.html', report=report)

@app.route('/documents')
def documents_page():
    index_status = get_index_status().to_dict()
    return render_template('documents.html', index_status=index_status)

@app.route('/api/health')
def health_api():
    return jsonify(get_health_report().to_dict())

@app.route('/api/documents')
def documents_api():
    return jsonify(get_index_status().to_dict())

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        state = conversation_store.get(get_session_id())
        result = answer_query_with_sources(query, conversation_state=state)
        payload = result.to_dict()
        payload["turn_count"] = state.turn_count()
        return jsonify(payload)
    except AssistantError as e:
        app.logger.warning("Chat request failed: %s", e)
        return jsonify({'error': str(e), 'code': e.code}), e.status_code
    except Exception:
        app.logger.exception("Unexpected chat request failure")
        return jsonify({'error': 'Unexpected error processing request.'}), 500

@app.route('/api/chat/session', methods=['DELETE'])
def clear_chat_session():
    conversation_store.clear(get_session_id())
    return jsonify({"status": "cleared", "turn_count": 0})

def get_session_id():
    if "chat_session_id" not in session:
        session["chat_session_id"] = uuid4().hex
    return session["chat_session_id"]

if __name__ == '__main__':
    app.run(debug=True)
