# OpenAI tool calling with history 
### Uses a sample function
import yaml
import json
import os
from openai import OpenAI

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "..", "character_config.yaml")

with open(file_path, 'r') as f:
    char_config = yaml.safe_load(f)

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key=char_config['api_key']
)

HISTORY_FILE = char_config['history_file']
MODEL = char_config['model']

# Combine system_prompt + emotions_prompt into one system message
_system_text = char_config['presets']['default']['system_prompt'].strip()
_emotions_text = char_config['presets']['default'].get('emotions_prompt', '').strip()
SYSTEM_PROMPT_TEXT = _system_text + "\n\n" + _emotions_text if _emotions_text else _system_text

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT_TEXT
    }
]

# Load/save chat history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        # Pastikan system prompt selalu ada di awal
        if not history or history[0].get("role") != "system":
            history = SYSTEM_PROMPT + history
        else:
            # Update system prompt jika berubah
            history[0]["content"] = SYSTEM_PROMPT_TEXT
        return history
    return SYSTEM_PROMPT

def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_response(user_input):
    messages = load_history()

    # Append user message to memory
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Use standard chat completions (compatible with Ollama)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        max_tokens=2048,
    )

    reply = response.choices[0].message.content

    messages.append({
        "role": "assistant",
        "content": reply
    })

    save_history(messages)
    return reply

if __name__ == "__main__":
    print('running main')
