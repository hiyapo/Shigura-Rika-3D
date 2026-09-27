"""
VRM Bridge — WebSocket server yang menghubungkan main.py ke browser VRM client.
Mengirim LLM response + detected emotion ke browser untuk facial expression & lip sync.
"""

import asyncio
import json
import threading
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

# Connected WebSocket clients
connected_clients: list[WebSocket] = []

# Valid VRM expression keywords
VALID_EXPRESSIONS = {"happy", "sad", "angry", "surprised", "neutral"}

# ============================================================
# Expression Tag Parser
# ============================================================

def parse_expression_segments(text: str) -> list[dict]:
    """
    Parse LLM output that contains [emotion] tags into a list of segments.
    Each segment has a 'text' (what to say) and 'emotion' (expression to show).
    Any unrecognized bracket tags (e.g. [smiling playfully]) are stripped from the text.

    Example input: "[happy] Yay! [surprised] Wait, really?! [angry] Ugh!"
    Example output: [
        {"text": "Yay!", "emotion": "happy"},
        {"text": "Wait, really?!", "emotion": "surprised"},
        {"text": "Ugh!", "emotion": "angry"},
    ]
    """
    # Regex: match ANY bracket tag, including multi-word like [smiling playfully]
    tag_pattern = re.compile(r'\[([^\]]+)\]')

    segments = []
    parts = tag_pattern.split(text)
    # parts alternates: [text_before_first_tag, tag1, text1, tag2, text2, ...]

    current_emotion = "neutral"  # Start with neutral if no tag at the beginning

    i = 0
    while i < len(parts):
        chunk = parts[i].strip()

        if i + 1 < len(parts):
            # Next item is a tag
            next_tag = parts[i + 1].strip().lower()
            if next_tag in VALID_EXPRESSIONS:
                # Valid emotion tag — emit previous text with current emotion, then switch
                if chunk:
                    segments.append({"text": chunk, "emotion": current_emotion})
                current_emotion = next_tag
            else:
                # Unrecognized tag (e.g. [smiling playfully], [giggles]) — strip it, keep text
                if chunk:
                    segments.append({"text": chunk, "emotion": current_emotion})
                print(f"[VRM Bridge] Stripped unknown tag: [{next_tag}]")
            i += 2  # always skip the tag
        else:
            # Last text segment (no more tags)
            if chunk:
                segments.append({"text": chunk, "emotion": current_emotion})
            i += 1

    # If nothing parsed, return as single neutral segment
    if not segments:
        clean = tag_pattern.sub('', text).strip()
        segments.append({"text": clean or text, "emotion": "neutral"})

    return segments


def strip_expression_tags(text: str) -> str:
    """Remove ALL bracket tags from text (for TTS input), including multi-word ones like [smiling playfully]."""
    return re.sub(r'\[[^\]]+\]', '', text).strip()



def detect_emotion(text: str) -> str:
    """Detect dominant emotion from expression tags in text, fallback to neutral."""
    tag_pattern = re.compile(r'\[(\w+)\]')
    tags = [m.lower() for m in tag_pattern.findall(text)]
    valid_tags = [t for t in tags if t in VALID_EXPRESSIONS and t != "neutral"]
    return valid_tags[0] if valid_tags else "neutral"


# ============================================================
# WebSocket Endpoint
# ============================================================

import queue
client_message_queue = queue.Queue()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[VRM Bridge] Client connected. Total clients: {len(connected_clients)}")

    try:
        import uuid
        from pathlib import Path
        
        while True:
            # Receive messages from the browser (text or binary audio)
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                print("[VRM Bridge] Client disconnected.")
                break
            
            if "text" in message:
                data = message["text"]
                print(f"[VRM Bridge] Received text from client: {data}")
                try:
                    msg = json.loads(data)
                    if "text" in msg:
                        client_message_queue.put(msg)
                except json.JSONDecodeError:
                    client_message_queue.put({"text": data})
                    
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                print(f"[VRM Bridge] Received {len(audio_bytes)} bytes of audio from client.")
                
                try:
                    # Detect format from magic bytes
                    if audio_bytes[:4] == b'\x1aE\xdf\xa3' or audio_bytes[:4] == b'\x1aE\xdf\xa3':
                        ext = "webm"
                    elif audio_bytes[:4] == b'OggS':
                        ext = "ogg"
                    elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                        ext = "mp3"
                    elif audio_bytes[:4] == b'RIFF':
                        ext = "wav"
                    else:
                        # Check for WebM (starts with 0x1A 0x45 0xDF 0xA3)
                        if len(audio_bytes) > 4 and audio_bytes[0] == 0x1A and audio_bytes[1] == 0x45:
                            ext = "webm"
                        else:
                            ext = "webm"  # Default fallback
                    
                    audio_path = Path("func") / "audio" / f"browser_audio.{ext}"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"[VRM Bridge] Audio saved to: {audio_path.resolve()}")
                    
                    # Notify main.py that an audio file is ready to be transcribed
                    client_message_queue.put({
                        "type": "audio",
                        "file": str(audio_path.resolve()),
                        "ext": ext,
                    })
                except Exception as e:
                    print(f"[VRM Bridge] Error saving audio: {e}")
                
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"[VRM Bridge] Client disconnected. Total clients: {len(connected_clients)}")


async def _broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    if not connected_clients:
        print("[VRM Bridge] No clients connected, skipping broadcast.")
        return

    payload = json.dumps(message)
    disconnected = []

    for client in connected_clients:
        try:
            await client.send_text(payload)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        connected_clients.remove(client)


def send_to_vrm(text: str, emotion: str = None):
    """
    Send a message to the VRM client (called from main.py).
    Parses [emotion] tags from the LLM output into per-segment expressions.
    """
    segments = parse_expression_segments(text)

    # Build a clean_text (no tags) for display
    clean_text = strip_expression_tags(text)

    # speak_sequence: browser will show each segment with its emotion
    message = {
        "type": "speak_sequence",
        "clean_text": clean_text,
        "segments": segments,     # [{"text": "...", "emotion": "happy"}, ...]
    }

    print(f"[VRM Bridge] Sending speak_sequence with {len(segments)} segment(s):")
    for seg in segments:
        print(f"   [{seg['emotion']}] {seg['text'][:60]}")

    # Schedule the broadcast on the event loop
    try:
        loop = _server_loop
        asyncio.run_coroutine_threadsafe(_broadcast(message), loop)
    except Exception as e:
        print(f"[VRM Bridge] Failed to send: {e}")


def send_emotion_only(emotion: str):
    """Send just an emotion change without speech (e.g., for idle state)."""
    message = {
        "type": "emotion",
        "emotion": emotion,
    }
    try:
        loop = _server_loop
        asyncio.run_coroutine_threadsafe(_broadcast(message), loop)
    except Exception as e:
        print(f"[VRM Bridge] Failed to send emotion: {e}")


# ============================================================
# Server Lifecycle
# ============================================================

_server_loop = None


def _run_server(host: str, port: int):
    """Run the uvicorn server in a new event loop (for threading)."""
    global _server_loop
    _server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_server_loop)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    _server_loop.run_until_complete(server.serve())


def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the WebSocket server in a background daemon thread."""
    thread = threading.Thread(
        target=_run_server,
        args=(host, port),
        daemon=True,
        name="vrm-bridge"
    )
    thread.start()
    print(f"\n[VRM Bridge] WebSocket server started on ws://{host}:{port}/ws\n")
    return thread
