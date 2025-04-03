import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import yaml # For config loading
from dotenv import load_dotenv # Import load_dotenv
import os
from typing import List, Dict, Optional
import torch # Needed for Silero VAD

# Import custom modules
from src.stt_module import SpeechToText
from src.llm_module import LanguageModel
from src.tts_module import TextToSpeech
from src.rag_module import RAGManager # Import RAGManager
from src.gui import VoiceAgentApp # Import the GUI class

# TODO: Import chosen VAD library based on config[\'audio\'][\'vad_implementation\']
# Example for Silero:
# import torch
# if config[\'audio\'][\'vad_implementation\'] == \'silero\':
#     try:
#         # torch.set_num_threads(1) # Consider setting globally if needed
#         vad_model, vad_utils = torch.hub.load(repo_or_dir=\'snakers4/silero-vad\',
#                                             model=\'silero_vad\',
#                                             force_reload=False) 
#         (get_speech_timestamps,
#         save_audio,
#         read_audio,
#         VADIterator,
#         collect_chunks) = vad_utils
#         logger.info("Silero VAD model loaded.")
#     except Exception as e:
#         logger.exception(f"Error loading Silero VAD model: {e}")
#         vad_model = None
# else: # Add other VAD implementations here
#     vad_model = None
#     VADIterator = None # Ensure VADIterator is defined or None


# --- Configuration Loading ---
load_dotenv() # Load .env file
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {CONFIG_PATH}")
except FileNotFoundError:
    logger.error(f"Configuration file not found at {CONFIG_PATH}. Exiting.")
    exit(1)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    exit(1)

# --- Logging Setup ---
log_level = config.get('log_level', 'INFO').upper()
logging.basicConfig(level=log_level, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Log level set to {log_level}")


# --- Calculate derived audio settings ---
SAMPLE_RATE = config['audio']['sample_rate']
CHUNK_DURATION_MS = config['audio']['chunk_duration_ms']
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
logger.info(f"Audio Settings: Sample Rate={SAMPLE_RATE}Hz, Chunk Size={CHUNK_SIZE} frames ({CHUNK_DURATION_MS}ms)")

# --- Global State & Queues ---
app: Optional[VoiceAgentApp] = None # Global reference to the GUI app
audio_input_queue = queue.Queue() # Queue for raw audio chunks from mic
audio_playback_queue = queue.Queue() # Queue for synthesized audio chunks to play
playback_thread: Optional[threading.Thread] = None
audio_processor_thread: Optional[threading.Thread] = None
input_stream: Optional[sd.InputStream] = None
speech_buffer = []
is_speaking = False
is_agent_speaking = False
last_speech_time = time.time()
conversation_history = []
processing_thread: Optional[threading.Thread] = None # Thread for the actual pipeline
shutdown_event = threading.Event()
current_emotion_instruction: Optional[str] = None # Placeholder for GUI input

# --- Module Initialization ---
stt_model: Optional[SpeechToText] = None
llm_model: Optional[LanguageModel] = None
tts_model: Optional[TextToSpeech] = None
rag_manager: Optional[RAGManager] = None
vad_iterator = None

# --- Load Silero VAD ---
VAD_MODEL_INSTANCE = None
VAD_UTILS = None
VADIterator = None
try:
    # Reduce threads for VAD model loading if needed
    # torch.set_num_threads(1)
    vad_model_instance, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = vad_utils
    logger.info("Silero VAD model loaded successfully.")
    VAD_MODEL_INSTANCE = vad_model_instance # Store the model instance
    VAD_UTILS = vad_utils # Store utils if needed elsewhere
except Exception as e:
    logger.exception(f"Error loading Silero VAD model: {e}. VAD will not function.")
    # VADIterator remains None

# --- Audio Callback ---
def audio_callback(indata, frames, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        logger.warning(f"Audio callback status: {status}")
    if not shutdown_event.is_set():
        audio_input_queue.put(indata.copy()) # Use audio_input_queue

# --- VAD & Processing Logic ---
def process_audio_stream():
    global is_speaking, last_speech_time, speech_buffer, processing_thread, is_agent_speaking, app

    logger.info("Audio processing thread started.")
    if app: app.update_status("Listening...") # Initial status
    min_silence_duration_s = config['audio']['silence_duration_ms'] / 1000.0
    max_speech_duration_s = config['audio']['max_speech_duration_s']
    post_playback_delay_s = config['audio']['post_playback_delay_ms'] / 1000.0
    min_silence_chunks = int(min_silence_duration_s * SAMPLE_RATE / CHUNK_SIZE)
    max_speech_chunks = int(max_speech_duration_s * SAMPLE_RATE / CHUNK_SIZE)
    vad_threshold = config['audio']['vad_threshold']
    silence_counter = 0

    while not shutdown_event.is_set():
        try:
            audio_chunk = audio_input_queue.get(timeout=0.1)

            # --- Agent Speaking Delay ---
            # If agent just finished speaking, wait a bit before processing user audio
            if is_agent_speaking:
                continue # Skip processing user audio while agent might still be audible
            if time.time() < last_speech_time + post_playback_delay_s and not is_speaking:
                 continue # Still in post-playback delay

            is_currently_speech = False
            if vad_iterator:
                try:
                    # Silero VAD expects a Float32 Tensor
                    audio_float32 = torch.from_numpy(audio_chunk).float()
                    # vad_iterator returns speech chunks boundaries
                    speech_dict = vad_iterator(audio_float32, return_seconds=False) # Use return_seconds=False
                    if speech_dict:
                        # Simple check: if 'start' or 'end' is detected in the chunk
                        is_currently_speech = True
                except Exception as vad_e:
                    logger.error(f"Error during Silero VAD processing: {vad_e}")
                    is_currently_speech = False # Fallback on error
                    # Maybe disable VAD after repeated errors?
                speech_buffer.append(audio_chunk) # Store recent chunks
                if len(speech_buffer) > max_speech_chunks: # Keep buffer short
                    speech_buffer.pop(0)
            else:
                # VAD Disabled - Assume always speaking or implement volume check
                # logger.warning("VAD not active.")
                is_currently_speech = True # Simple fallback: process everything if no VAD
                speech_buffer = [audio_chunk] # No padding needed if no VAD

            # --- Interruption / Speech Handling ---
            if is_currently_speech:
                if is_agent_speaking:
                    logger.info("User interruption detected!")
                    if app: app.update_status("Interrupted! Listening...")
                    is_agent_speaking = False
                    # Clear the playback queue
                    while not audio_playback_queue.empty():
                        try: audio_playback_queue.get_nowait() 
                        except queue.Empty: break
                    sd.stop() # Stop current playback if any
                    # Signal LLM to stop? (Best effort - vLLM might not support this easily)
                    # We might need to add cancellation support to the LLM module later.

                    # Start processing interruption immediately
                    speech_buffer = speech_buffer[-max_speech_chunks:] + [audio_chunk] # Use padding
                    is_speaking = True
                    last_speech_time = time.time()
                    silence_counter = 0
                elif not is_speaking:
                    logger.debug("Speech start detected.")
                    if app: app.update_status("User speaking...")
                    is_speaking = True
                    # Include padding chunks before the speech chunk
                    speech_buffer = speech_buffer[-max_speech_chunks:] + [audio_chunk]
                    is_speaking = True
                    last_speech_time = time.time()
                    silence_counter = 0
                else:
                    # Continue ongoing user speech
                    speech_buffer.append(audio_chunk)
                    last_speech_time = time.time()
                    silence_counter = 0

                # Force processing if speech is too long
                if is_speaking and len(speech_buffer) > max_speech_chunks:
                    logger.warning(f"Maximum speech duration ({max_speech_duration_s}s) reached. Forcing processing.")
                    if not (processing_thread and processing_thread.is_alive()):
                        process_user_speech(list(speech_buffer)) # Process current buffer
                    # Reset state after forcing process
                    speech_buffer = [] 
                    is_speaking = False
                    silence_counter = 0
            
            # --- Silence Handling ---
            else: # Not speech
                if is_speaking:
                    # Append non-speech chunk to buffer briefly for trailing silence context
                    speech_buffer.append(audio_chunk)
                    silence_counter += 1
                    if silence_counter >= min_silence_chunks:
                        logger.debug(f"End of speech detected by silence ({min_silence_duration_s:.2f}s).")
                        if app: app.update_status("Processing speech...") # Update status before processing
                        if speech_buffer and not (processing_thread and processing_thread.is_alive()):
                            process_user_speech(list(speech_buffer))
                        speech_buffer = []
                        is_speaking = False
                        silence_counter = 0
                elif not is_agent_speaking and app and app.status_label.cget("text") != "Status: Listening...": # Only update if not already listening
                      if time.time() > last_speech_time + post_playback_delay_s: # Ensure post-playback delay passed
                           # Check processing thread status before setting to Listening
                           if not (processing_thread and processing_thread.is_alive()):
                                app.update_status("Listening...")
                # else: User is not speaking, and wasn't speaking before - reset status if needed

            # Reset VAD state for next chunk if needed by iterator implementation
            if vad_iterator and hasattr(vad_iterator, 'reset_states'):
                 vad_iterator.reset_states()

        except queue.Empty:
            # Timeout handling
            if is_speaking and (time.time() - last_speech_time) > min_silence_duration_s:
                 logger.debug(f"End of speech detected by timeout ({min_silence_duration_s:.2f}s).")
                 if app: app.update_status("Processing speech...")
                 if speech_buffer and not (processing_thread and processing_thread.is_alive()):
                     process_user_speech(list(speech_buffer))
                 speech_buffer = []
                 is_speaking = False
                 silence_counter = 0
            continue
        except Exception as e:
            logger.exception(f"Error in audio processing loop: {e}")
            if app: app.update_status("Error processing audio")
            time.sleep(0.1)

    logger.info("Audio processing thread finished.")


def process_user_speech(audio_chunks: List[np.ndarray]):
    """Handles the STT -> RAG -> LLM -> TTS pipeline in a separate thread."""
    global processing_thread, app
    # Ensure only one pipeline runs at a time
    if processing_thread and processing_thread.is_alive():
        logger.warning("Processing thread already running. Skipping new request.")
        return
    if app: app.update_status("Thinking...") # Update status
    processing_thread = threading.Thread(target=run_inference_pipeline, args=(audio_chunks,))
    processing_thread.start()

def run_inference_pipeline(audio_chunks: List[np.ndarray]):
    """The function executed in the processing thread."""
    global conversation_history, is_agent_speaking, last_speech_time, app

    if not audio_chunks:
        logger.warning("Tried to process empty audio buffer.")
        return

    if stt_model is None or llm_model is None or tts_model is None:
        logger.error("Core modules (STT, LLM, TTS) not initialized. Cannot run pipeline.")
        return

    start_pipeline_time = time.time()
    logger.info(f"Pipeline started for {len(audio_chunks) * CHUNK_DURATION_MS / 1000.0:.2f}s of audio.")

    try:
        # 1. STT
        full_audio = np.concatenate(audio_chunks).flatten()
        stt_start_time = time.time()
        user_text = stt_model.transcribe_audio(full_audio, SAMPLE_RATE)
        stt_time = time.time() - stt_start_time
        if not user_text:
            logger.warning(f"STT resulted in empty text ({stt_time:.2f}s). Skipping LLM call.")
            return
        logger.info(f"STT ({stt_time:.2f}s): {user_text}")

        # 2. Update History & Pruning
        conversation_history.append({"role": "user", "content": user_text})
        max_turns = config.get('general', {}).get('max_history_turns', 10)
        if len(conversation_history) > max_turns * 2: # Each turn has user + assistant
            logger.debug(f"Pruning conversation history from {len(conversation_history)//2} turns to {max_turns} turns.")
            # Keep the system prompt (if we add one) + the last max_turns*2 messages
            conversation_history = conversation_history[-(max_turns*2):]

        # 3. RAG Retrieval
        retrieved_docs = []
        rag_time = 0
        if rag_manager:
            rag_start_time = time.time()
            retrieved_docs = rag_manager.retrieve(query=user_text)
            rag_time = time.time() - rag_start_time
            if retrieved_docs:
                logger.info(f"RAG ({rag_time:.2f}s): Retrieved {len(retrieved_docs)} documents.")
                # logger.debug(f"Retrieved docs: {retrieved_docs}") # Can be verbose
            else:
                 logger.info(f"RAG ({rag_time:.2f}s): No relevant documents found.")

        # 4. LLM Inference (Streaming)
        llm_start_time = time.time()
        logger.info("Starting LLM stream generation...")
        llm_stream = llm_model.generate_response_stream(
            history=conversation_history,
            retrieved_docs=retrieved_docs,
            emotion_instruction=current_emotion_instruction, # Get from GUI state later
            **config.get('llm', {}).get('generation', {}) # Pass generation kwargs
        )

        # 5. Consume LLM Stream to get full response
        agent_response_text = ""
        llm_first_token_time = -1
        for chunk in llm_stream:
            if llm_first_token_time < 0:
                 llm_first_token_time = time.time() - llm_start_time
                 logger.info(f"LLM time to first token: {llm_first_token_time:.2f}s")
            # print(chunk, end="", flush=True) # Debug: print LLM chunks
            agent_response_text += chunk

        llm_total_time = time.time() - llm_start_time
        if not agent_response_text:
             logger.warning(f"LLM ({llm_total_time:.2f}s) generated empty response.")
             return # Don't proceed with TTS if response is empty
        logger.info(f"LLM ({llm_total_time:.2f}s) full response: {agent_response_text[:100]}...")

        # 6. Update History with Assistant Response
        conversation_history.append({"role": "assistant", "content": agent_response_text})
        # Apply pruning again after assistant response if needed (optional)
        if len(conversation_history) > max_turns * 2:
             conversation_history = conversation_history[-(max_turns*2):]

        # 7. TTS Synthesis (Non-streaming)
        tts_start_time = time.time()
        logger.info("Synthesizing speech with TTS...")
        # TODO: Add logic to create TTS context segments if needed for voice cloning/prompting
        tts_context = []
        audio_data, tts_sr = tts_model.synthesize_speech(
            text=agent_response_text,
            context=tts_context
            # Pass other TTS params from config if needed
        )
        tts_time = time.time() - tts_start_time

        if audio_data is None or audio_data.size == 0:
            logger.error(f"TTS ({tts_time:.2f}s) failed to generate audio.")
            return
        logger.info(f"TTS ({tts_time:.2f}s) synthesized {len(audio_data)/tts_sr:.2f}s of audio.")

        # 8. Queue Audio for Playback
        # Put the entire audio data and its sample rate onto the playback queue
        audio_playback_queue.put((audio_data, tts_sr))

    except Exception as e:
        logger.exception("Error occurred during inference pipeline")
    finally:
        pipeline_duration = time.time() - start_pipeline_time
        logger.info(f"Inference pipeline finished in {pipeline_duration:.2f}s.")
        # The is_agent_speaking flag is handled by the playback thread now

# --- Audio Playback Thread ---
def playback_worker():
    global is_agent_speaking, last_speech_time
    logger.info("Audio playback thread started.")
    while not shutdown_event.is_set():
        try:
            audio_data, sample_rate = audio_playback_queue.get(timeout=0.1)
            
            logger.info(f"Playing {len(audio_data)/sample_rate:.2f}s of synthesized audio (SR: {sample_rate}Hz)...")
            is_agent_speaking = True
            try:
                sd.play(audio_data, sample_rate, blocking=True) # Play synchronously for now
                # If using blocking=False, need a callback or check mechanism:
                # stream = sd.play(audio_data, sample_rate, blocking=False)
                # while stream.active and not shutdown_event.is_set() and not audio_playback_queue.empty(): # Check for interruption
                #     time.sleep(0.05)
                # sd.stop() # Ensure stop on interrupt or finish
                logger.info("Playback finished.")
            except Exception as play_e:
                logger.exception(f"Error during sounddevice playback: {play_e}")
            finally:
                is_agent_speaking = False
                last_speech_time = time.time() # Record time agent finished speaking
                audio_playback_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
             logger.exception(f"Error in playback worker: {e}")
             time.sleep(0.1)
    logger.info("Audio playback thread finished.")

# --- Main Execution ---
if __name__ == "__main__":
    if stt_model is None or llm_model is None or tts_model is None:
         logger.error("Core modules failed to initialize. Exiting.")
         exit(1)
         
    # Start the audio processing thread
    audio_processor = threading.Thread(target=process_audio_stream, daemon=True)
    audio_processor.start()

    # Start the audio playback thread
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    # Start the audio input stream
    try:
        logger.info("Starting audio input stream...")
        with sd.InputStream(samplerate=SAMPLE_RATE, 
                           blocksize=CHUNK_SIZE, 
                           device=config['audio'].get('input_device_index', None),
                           channels=1, 
                           dtype='float32', 
                           callback=audio_callback):
            logger.info("Application started. Press Ctrl+C to exit.")
            while not shutdown_event.is_set():
                time.sleep(0.5) # Keep main thread alive

    except KeyboardInterrupt:
        logger.info("Shutdown signal received (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"An error occurred in the main loop: {e}")
    finally:
        logger.info("Shutting down...")
        shutdown_event.set()
        
        # Wait for threads to finish (with timeout)
        logger.info("Waiting for audio processor thread...")
        audio_processor.join(timeout=2.0)
        logger.info("Waiting for playback thread...")
        # Signal playback queue to stop if needed
        if playback_thread is not None:
            audio_playback_queue.put(None) # Sentinel value to stop worker
            playback_thread.join(timeout=2.0)
            
        if processing_thread and processing_thread.is_alive():
            logger.info("Waiting for active inference pipeline thread...")
            processing_thread.join(timeout=5.0) # Wait longer for pipeline

        logger.info("Shutdown complete.") 