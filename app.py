import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
PROJECT_DIR = Path(__file__).resolve().parent
CHAT_DIR = PROJECT_DIR / "chats"
MEMORY_FILE = PROJECT_DIR / "memory.json"
TEST_MESSAGE = "Hello!"


st.set_page_config(page_title="My AI Chat", layout="wide")


def utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def format_timestamp(timestamp):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return timestamp


def chat_file_path(chat_id):
    return CHAT_DIR / f"{chat_id}.json"


def build_chat_title(messages):
    for message in messages:
        if message["role"] == "user":
            words = " ".join(message["content"].split())
            if not words:
                break
            return words[:40] + ("..." if len(words) > 40 else "")
    return "New Chat"


def create_chat(initial_messages=None):
    timestamp = utc_now_iso()
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": initial_messages or [],
    }


def normalize_chat(chat):
    chat.setdefault("id", str(uuid.uuid4()))
    chat.setdefault("created_at", utc_now_iso())
    chat.setdefault("updated_at", chat["created_at"])
    chat["messages"] = chat.get("messages", [])
    chat["title"] = chat.get("title") or build_chat_title(chat["messages"])
    return chat


def migrate_chat_files():
    CHAT_DIR.mkdir(exist_ok=True)
    misplaced_chat = CHAT_DIR / "memory.json"
    if not misplaced_chat.exists():
        return

    try:
        data = json.loads(misplaced_chat.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(data, dict) or "messages" not in data:
        return

    chat = normalize_chat(data)
    correct_path = chat_file_path(chat["id"])
    if not correct_path.exists():
        correct_path.write_text(json.dumps(chat, indent=2), encoding="utf-8")
    misplaced_chat.unlink()


def save_chat(chat):
    CHAT_DIR.mkdir(exist_ok=True)
    chat["title"] = build_chat_title(chat["messages"])
    chat["updated_at"] = utc_now_iso()
    chat_file_path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_saved_chats():
    CHAT_DIR.mkdir(exist_ok=True)
    chats = []
    for path in sorted(CHAT_DIR.glob("*.json")):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
            chats.append(normalize_chat(chat))
        except (OSError, json.JSONDecodeError):
            continue

    chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    return chats


def load_memory():
    if not MEMORY_FILE.exists():
        return {}

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_memory(memory):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def clear_memory():
    st.session_state.memory = {}
    save_memory({})


def merge_memory(existing_memory, new_memory):
    merged = dict(existing_memory)
    for key, value in new_memory.items():
        if value in ("", None, [], {}):
            continue
        merged[key] = value
    return merged


def build_system_prompt(memory):
    if not memory:
        return "You are a helpful, friendly AI assistant. Answer clearly and naturally."

    memory_lines = [f"- {key}: {value}" for key, value in memory.items()]
    return (
        "You are a helpful, friendly AI assistant. Use the saved user memory below "
        "to personalize your responses when relevant, but do not mention the memory "
        "unless it naturally helps the conversation.\n\nUser memory:\n"
        + "\n".join(memory_lines)
    )


def build_model_messages(messages, memory):
    return [{"role": "system", "content": build_system_prompt(memory)}, *messages]


def extract_user_memory(user_message, hf_token):
    prompt = (
        "Given the user message below, extract any stable personal facts, preferences, "
        "or traits as a JSON object. Keep only useful memory such as name, preferred "
        "language, interests, favorite topics, dislikes, goals, or communication "
        "style. If there is nothing useful to store, return {} only.\n\n"
        f"User message: {user_message}"
    )

    response_text = request_chat_completion(
        [{"role": "user", "content": prompt}],
        hf_token,
    )

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    return data if isinstance(data, dict) else {}


def get_hf_token():
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    if not token or token == "your_token_here":
        return None
    return token


def request_chat_completion(messages, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = response.status_code
        if status == 401:
            raise RuntimeError("The Hugging Face token is invalid or unauthorized.") from exc
        if status == 429:
            raise RuntimeError("The API rate limit was reached. Please try again soon.") from exc
        raise RuntimeError(f"The API returned an error ({status}).") from exc

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The API returned an unexpected response.") from exc


def stream_chat_completion(messages, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=30,
        stream=True,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = response.status_code
        if status == 401:
            raise RuntimeError("The Hugging Face token is invalid or unauthorized.") from exc
        if status == 429:
            raise RuntimeError("The API rate limit was reached. Please try again soon.") from exc
        raise RuntimeError(f"The API returned an error ({status}).") from exc

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("The API returned an unexpected streamed response.") from exc

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            yield content
            time.sleep(0.02)


def ensure_state():
    migrate_chat_files()
    if "chats" not in st.session_state:
        st.session_state.chats = load_saved_chats()
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = (
            st.session_state.chats[0]["id"] if st.session_state.chats else None
        )
    if "api_tested" not in st.session_state:
        st.session_state.api_tested = False
    if "api_test_response" not in st.session_state:
        st.session_state.api_test_response = None
    if "api_test_error" not in st.session_state:
        st.session_state.api_test_error = None
    if "memory" not in st.session_state:
        st.session_state.memory = load_memory()
        if not MEMORY_FILE.exists():
            save_memory(st.session_state.memory)


def get_active_chat():
    active_chat_id = st.session_state.active_chat_id
    for chat in st.session_state.chats:
        if chat["id"] == active_chat_id:
            return chat
    return None


def set_active_chat(chat_id):
    st.session_state.active_chat_id = chat_id


def add_new_chat():
    chat = create_chat()
    st.session_state.chats.insert(0, chat)
    st.session_state.active_chat_id = chat["id"]
    save_chat(chat)


def delete_chat(chat_id):
    st.session_state.chats = [chat for chat in st.session_state.chats if chat["id"] != chat_id]
    path = chat_file_path(chat_id)
    if path.exists():
        path.unlink()

    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = (
            st.session_state.chats[0]["id"] if st.session_state.chats else None
        )


def run_initial_api_test(hf_token):
    if st.session_state.api_tested:
        return

    st.session_state.api_tested = True
    st.session_state.api_test_error = None
    st.session_state.api_test_response = None

    try:
        st.session_state.api_test_response = request_chat_completion(
            build_model_messages(
                [{"role": "user", "content": TEST_MESSAGE}],
                st.session_state.memory,
            ),
            hf_token,
        )
    except requests.RequestException:
        st.session_state.api_test_error = (
            "Network error while contacting Hugging Face. Please try again."
        )
    except RuntimeError as exc:
        st.session_state.api_test_error = str(exc)


def render_sidebar():
    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat", use_container_width=True):
            add_new_chat()
            st.rerun()

        with st.expander("User Memory", expanded=True):
            if st.session_state.memory:
                st.json(st.session_state.memory)
            else:
                st.caption("No saved user memory yet.")
            if st.button("Clear Memory", use_container_width=True):
                clear_memory()
                st.rerun()

        if not st.session_state.chats:
            st.caption("No saved chats yet.")
            return

        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            row = st.container()
            with row:
                title_col, delete_col = st.columns([5, 1])
                label = f"{chat['title']}\n{format_timestamp(chat['updated_at'])}"
                if title_col.button(
                    label,
                    key=f"open_{chat['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    set_active_chat(chat["id"])
                    st.rerun()
                if delete_col.button("✕", key=f"delete_{chat['id']}"):
                    delete_chat(chat["id"])
                    st.rerun()


def render_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def append_assistant_message(chat, content):
    chat["messages"].append({"role": "assistant", "content": content})
    save_chat(chat)


def handle_chat_input(chat, hf_token):
    prompt = st.chat_input("Type your message here")
    if not prompt:
        return

    chat["messages"].append({"role": "user", "content": prompt})
    save_chat(chat)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_chat_completion(
                    build_model_messages(chat["messages"], st.session_state.memory),
                    hf_token,
                )
            )
        except requests.RequestException:
            reply = "Network error while contacting Hugging Face. Please try again."
            st.write(reply)
        except RuntimeError as exc:
            reply = str(exc)
            st.write(reply)

    append_assistant_message(chat, reply)

    try:
        extracted_memory = extract_user_memory(prompt, hf_token)
    except (requests.RequestException, RuntimeError):
        extracted_memory = {}

    if extracted_memory:
        st.session_state.memory = merge_memory(st.session_state.memory, extracted_memory)
        save_memory(st.session_state.memory)

    st.rerun()


def main():
    ensure_state()
    render_sidebar()

    st.title("My AI Chat")
    st.caption("A Streamlit chat app powered by the Hugging Face Inference Router.")

    hf_token = get_hf_token()
    if hf_token is None:
        st.error(
            "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml before "
            "running the app."
        )
        st.stop()

    run_initial_api_test(hf_token)
    if st.session_state.api_test_error:
        st.error(st.session_state.api_test_error)
    elif st.session_state.api_test_response:
        with st.expander("Part A API test reply", expanded=False):
            st.write(f'User: "{TEST_MESSAGE}"')
            st.write(st.session_state.api_test_response)

    active_chat = get_active_chat()
    if active_chat is None:
        st.info("Create a new chat from the sidebar to get started.")
        return

    render_messages(active_chat["messages"])
    handle_chat_input(active_chat, hf_token)


if __name__ == "__main__":
    main()
