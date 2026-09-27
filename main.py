from faster_whisper import WhisperModel
from func.recorder import record_and_transcribe
from func.model import get_response
from func.speech import sovits_gen, play_audio
from func.vrm_bridge import start_server as start_vrm_server, send_to_vrm, client_message_queue, strip_expression_tags
from pathlib import Path
import os
import time

### transcribe audio 
import uuid
import soundfile as sf

def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate

print(' \n ========= Starting Chat... ================ \n')

# Start VRM WebSocket bridge server (runs in background thread)
start_vrm_server(host="0.0.0.0", port=8765)
time.sleep(1)  # Give server a moment to start

whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")

while True:
    print("\n⏳ Waiting for text or audio input from Browser (Test Panel)...")
    
    # Wait for a message from the browser queue (blocking until a message arrives)
    user_msg = client_message_queue.get()
    
    user_spoken_text = ""
    
    # If the browser sent an audio file, transcribe it with Whisper!
    if user_msg.get("type") == "audio":
        audio_file = user_msg.get("file")
        ext = user_msg.get("ext", "?")
        print(f"🎙️ Received audio file: {audio_file} (format: {ext})")
        
        try:
            # faster-whisper can handle webm/ogg via ffmpeg automatically
            segments, info = whisper_model.transcribe(audio_file, beam_size=5)
            user_spoken_text = " ".join([segment.text for segment in segments]).strip()
            print(f"🗣️ Transcribed: '{user_spoken_text}' (lang: {info.language}, prob: {info.language_probability:.2f})")
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            print("   → Make sure ffmpeg is installed: https://ffmpeg.org/download.html")
            continue
        
    # If the browser sent text directly
    else:
        user_spoken_text = user_msg.get("text", "").strip()
        print(f"🗣️ User says: {user_spoken_text}")
        
    
    if not user_spoken_text:
        print("⚠️ Ignored empty input.")
        continue

    ### pass to LLM and get a LLM output.
    print("🤖 Thinking (waiting for LLM response)...")
    llm_output = get_response(user_spoken_text)
    tts_read_text = llm_output
    print(f"💬 LLM says: {tts_read_text}")

    # Clean expression tags for TTS (don't say "[happy]" out loud!)
    tts_clean_text = strip_expression_tags(tts_read_text)
    print(f"📢 TTS input (cleaned): {tts_clean_text}")

    ### file organization 
    # 1. Generate a unique filename
    uid = uuid.uuid4().hex
    filename = f"output_{uid}.wav"
    output_wav_path = Path("func") / f"./audio/{filename}"
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # generate audio from cleaned text (no tags)
    print("🔊 Generating audio (waiting for SoVITS TTS)...")
    gen_aud_path = sovits_gen(tts_clean_text, output_wav_path)

    ### Send expression sequence to VRM AFTER audio is ready!
    print("🌐 Sending expression sequence to VRM browser...")
    send_to_vrm(tts_read_text)  # Pass original text WITH tags for expression parsing

    if gen_aud_path is not None:
        print("▶️ Playing audio...")
        play_audio(gen_aud_path)
    else:
        print("[WARNING] sovits_gen fail, skip play_audio.")
        
    # clean up audio files
    [fp.unlink() for fp in Path("func/audio").glob("*.wav") if fp.is_file()]
    print("✅ Finished processing. Ready for next input.\n")