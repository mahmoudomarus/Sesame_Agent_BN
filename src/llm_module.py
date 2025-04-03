import logging
from typing import List, Dict, Any, Generator, Optional
import threading # Needed for potential Transformers streaming
import time
import os
from pathlib import Path
import sys # Moved import here

# Attempt to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None # Define as None if import fails
    SamplingParams = None # Define as None if import fails
    VLLM_AVAILABLE = False
    logging.warning("vLLM not found. Please install vLLM (`pip install vllm`) to use the LLM module.")

# Placeholder for actual LLM loading and inference
# This will depend heavily on the chosen inference engine (vLLM, llama-cpp-python, Transformers)

logger = logging.getLogger(__name__)

class LanguageModel:
    def __init__(self, model_path: str, engine: str = "vllm", device: str = "cuda", **engine_kwargs):
        """
        Initializes the LLM using vLLM.

        Args:
            model_path (str): Path to the downloaded LLM (Hugging Face identifier or local path).
            engine (str): Inference engine to use (should be 'vllm').
            device (str): Device for inference (vLLM typically manages this via CUDA).
            **engine_kwargs: Additional arguments passed directly to vLLM (e.g., gpu_memory_utilization, dtype).
        """
        logger.info(f"Initializing LLM from {model_path} using vLLM...")
        self.model_path = model_path
        self.engine = engine
        self.device = device # Stored for info, vLLM handles placement
        self.model: Optional[LLM] = None
        self.engine_kwargs = engine_kwargs

        if engine != "vllm":
            logger.error(f"This module currently only supports the 'vllm' engine, but '{engine}' was specified.")
            raise ValueError("Unsupported LLM engine. Only 'vllm' is implemented.")

        if not VLLM_AVAILABLE:
            logger.error("vLLM library is not available. Cannot initialize LLM.")
            # Optional: raise ImportError("vLLM is required but not installed.")
            return # Allow initialization but model will be None

        # Ensure model path exists if it's a local path
        if not model_path.startswith("models/") and "/" in model_path and not os.path.exists(model_path):
             logger.warning(f"Specified model_path '{model_path}' looks like a local path but does not exist.")
             # Consider raising an error or trying to treat it as HF identifier

        try:
            # vLLM initialization
            # Pass engine_kwargs directly to vLLM
            logger.info(f"Loading vLLM with args: {self.engine_kwargs}")
            self.model = LLM(model=model_path, **self.engine_kwargs)
            logger.info(f"vLLM loaded successfully for model: {model_path}")

        except Exception as e:
            logger.exception(f"Failed to initialize vLLM engine with model {model_path}: {e}")
            self.model = None # Ensure model is None on failure
            # Depending on requirements, might re-raise
            # raise e

    def generate_response(self, history: List[Dict[str, str]], retrieved_docs: List[str] = None, emotion_instruction: str = None, **generation_kwargs) -> str:
        """Generates a complete response (non-streaming) by consuming the stream."""
        response_chunks = []
        for chunk in self.generate_response_stream(history, retrieved_docs, emotion_instruction, **generation_kwargs):
            response_chunks.append(chunk)
        return "".join(response_chunks)

    def generate_response_stream(
        self,
        history: List[Dict[str, str]],
        retrieved_docs: List[str] = None,
        emotion_instruction: str = None,
        **generation_kwargs
    ) -> Generator[str, None, None]:
        """
        Generates a response stream using vLLM based on history, RAG context, and emotion.

        Args:
            history (List[Dict[str, str]]): Conversation history, formatted for the model.
                                           The last item should be the current user query.
            retrieved_docs (List[str], optional): List of document chunks retrieved by RAG.
            emotion_instruction (str, optional): Instruction for desired emotional tone.
            **generation_kwargs: Parameters for vLLM SamplingParams (e.g., max_tokens, temperature).

        Yields:
            str: Chunks of the generated text response.
        """
        if self.model is None or not VLLM_AVAILABLE:
            logger.warning("LLM model not loaded or vLLM unavailable. Yielding dummy response.")
            yield "Sorry, my language model is not available right now."
            return

        # 1. Format the input using the model's chat template logic
        formatted_prompt = self._format_input_chatml(history, retrieved_docs, emotion_instruction)

        # 2. Prepare Sampling Parameters
        # Map generation_kwargs from config to SamplingParams arguments
        # Note: 'max_new_tokens' in config becomes 'max_tokens' for SamplingParams
        sampling_kwargs = {
            "max_tokens": generation_kwargs.get("max_new_tokens", 150),
            "temperature": generation_kwargs.get("temperature", 0.7),
            "top_p": generation_kwargs.get("top_p", 0.9),
            "stop": generation_kwargs.get("stop_sequences", []), # Use 'stop' for vLLM
            # Add other relevant params like top_k if needed
        }
        if "top_k" in generation_kwargs:
             sampling_kwargs["top_k"] = generation_kwargs["top_k"]

        try:
            sampling_params = SamplingParams(**sampling_kwargs)
        except Exception as e:
            logger.exception(f"Failed to create vLLM SamplingParams with kwargs {sampling_kwargs}: {e}")
            yield "Sorry, there was an issue configuring the response generation."
            return

        logger.debug(f"Streaming response with vLLM. SamplingParams: {sampling_params}")
        # logger.debug(f"Formatted Prompt (last 500 chars): ...{formatted_prompt[-500:]}") # Can be very long

        start_time = time.time()
        first_token_yielded = False
        previous_text = ""

        try:
            # vLLM generate returns a generator of RequestOutput objects
            # We process one prompt at a time here.
            results_generator = self.model.generate([formatted_prompt], sampling_params)

            for request_output in results_generator:
                # Extract the text generated so far for our single prompt
                current_text = request_output.outputs[0].text

                # Calculate the newly generated part
                new_text = current_text[len(previous_text):]

                if new_text: # Only yield if there's new text
                    if not first_token_yielded:
                        ttft = time.time() - start_time
                        logger.info(f"LLM Time to first token (vLLM): {ttft:.2f}s")
                        first_token_yielded = True
                    yield new_text

                # Update the previously generated text
                previous_text = current_text

                # Check if generation finished (optional, vLLM handles stopping)
                # if request_output.finished:
                #     break

        except Exception as e:
            logger.exception("Exception during vLLM stream generation")
            yield " Sorry, an error occurred while generating the response. " # Added spaces for TTS buffer
        finally:
            # Log finish reason if available (might need specific vLLM API check)
            finish_reason = request_output.outputs[0].finish_reason if 'request_output' in locals() else 'unknown'
            total_time = time.time() - start_time
            logger.info(f"LLM stream finished in {total_time:.2f}s. Reason: {finish_reason}")

    def _format_input_chatml(
        self,
        history: List[Dict[str, str]],
        retrieved_docs: List[str] = None,
        emotion_instruction: str = None
    ) -> str:
        """Formats history and context using Phi-3's ChatML instruct format.

        Includes RAG context before the last user message and prepends emotion instruction
        to the system prompt.

        Args:
            history (List[Dict[str, str]]): Conversation history. Last item is the current user query.
            retrieved_docs (List[str], optional): List of document chunks from RAG.
            emotion_instruction (str, optional): Instruction for desired emotional tone.

        Returns:
            str: The fully formatted prompt string ready for the LLM.
        """
        # Phi-3 Instruct ChatML tokens
        SYS_START, SYS_END = "<|system|>", "<|end|>"
        USER_START, USER_END = "<|user|>", "<|end|>"
        ASS_START, ASS_END = "<|assistant|>", "<|end|>"
        EOS_TOKEN = "<|endoftext|>" # Or check tokenizer for actual EOS

        # Base system prompt
        system_prompt = "You are Bignoodle, a helpful, friendly, and informative AI voice assistant created by Tabi. Keep your responses concise and conversational."
        if emotion_instruction:
            system_prompt = f"{emotion_instruction.strip()} {system_prompt}"

        prompt_str = SYS_START + "\n" + system_prompt + SYS_END + "\n"

        # Process history, inserting RAG context before the last user message
        num_messages = len(history)
        for i, msg in enumerate(history):
            role = msg.get('role')
            content = msg.get('content', '')

            if role == 'user':
                # If this is the *last* user message, prepend RAG context if available
                if i == num_messages - 1 and retrieved_docs:
                    context_str = "\n\n".join(retrieved_docs)
                    content = f"Based on the following context:\n---\n{context_str}\n---\n\n{content}"
                prompt_str += USER_START + "\n" + content + USER_END + "\n"
            elif role == 'assistant':
                prompt_str += ASS_START + "\n" + content + ASS_END + "\n"
            else:
                 logger.warning(f"Unknown role in history: {role}")

        # Add the final assistant prompt start token to signal the model to respond
        prompt_str += ASS_START + "\n" # Add newline for clarity if model expects it

        # Note: vLLM typically handles adding EOS token based on sampling params,
        # but check Phi-3 requirements if issues arise. Do NOT add EOS_TOKEN here usually.

        return prompt_str

    # Keep a basic formatter as fallback or for other models
    def _format_input(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Basic helper to format prompt and history."""
        if history is None:
            history = []
        full_prompt = ""
        for msg in history:
            if msg.get('role') == 'user':
                full_prompt += f"User: {msg.get('content', '')}\n"
            elif msg.get('role') == 'assistant':
                full_prompt += f"Assistant: {msg.get('content', '')}\n"
        full_prompt += f"User: {prompt}\nAssistant: "
        return full_prompt

# Example usage (conceptual, updated for vLLM)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Use the actual model path from your downloads/config
    llm_model_path = "../models/Phi-3-medium-128k-instruct" # Adjust path if needed

    # Check if model path exists (relative to this file's location)
    script_dir = Path(__file__).parent
    absolute_model_path = (script_dir / llm_model_path).resolve()
    if not absolute_model_path.exists():
         logger.error(f"Model path '{absolute_model_path}' does not exist. Please check the path.")
         # Attempt to find relative path from project root
         project_root_path = script_dir.parent
         alt_path = (project_root_path / "models" / "Phi-3-medium-128k-instruct").resolve()
         if alt_path.exists():
             logger.info(f"Found alternative path: {alt_path}")
             llm_model_path = str(alt_path)
         else:
             logger.error("Could not find model path. Exiting example.")
             sys.exit(1)
    else:
        llm_model_path = str(absolute_model_path) # Use absolute path for vLLM if found


    # Engine Kwargs (mirroring config.yaml example)
    vllm_kwargs = {
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.7, # Lowered from 0.8
        "tensor_parallel_size": 1,
        "dtype": 'bfloat16',
        "max_model_len": 8192, # Added to limit max sequence length
    }

    # Generation Kwargs (mirroring config.yaml example)
    gen_kwargs = {
        "max_new_tokens": 100, # Smaller for testing
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["<|end|>", "\nUser:"] # Ensure correct stop tokens for Phi-3
    }
    # --- End Configuration ---


    if not VLLM_AVAILABLE:
        logger.error("Cannot run example: vLLM is not installed.")
        sys.exit(1)

    try:
        print("Initializing vLLM...")
        llm = LanguageModel(model_path=llm_model_path, engine="vllm", **vllm_kwargs)
        print("vLLM Initialized.")

        if llm.model is None:
             print("LLM model failed to load. Exiting.")
             sys.exit(1)

        # --- Test Case 1: Simple Conversation ---
        print("\n--- Test Case 1: Simple Conversation ---")
        test_history_1 = [
            {"role": "user", "content": "Hello there!"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "Tell me a short joke about computers."} # Last user message
        ]
        print("Streaming response...")
        full_response_1 = ""
        response_stream_1 = llm.generate_response_stream(history=test_history_1, **gen_kwargs)
        for chunk in response_stream_1:
            print(chunk, end="", flush=True)
            full_response_1 += chunk
        print("\n--- End Test Case 1 ---")


        # --- Test Case 2: Conversation with RAG ---
        print("\n--- Test Case 2: Conversation with RAG ---")
        test_history_2 = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What does the retrieved document say about its population?"} # Last user message
        ]
        rag_docs = [
            "Document 1: Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its population is slightly over 2 million.",
            "Document 2: The Eiffel Tower is an iconic landmark in Paris."
        ]
        print("Streaming response with RAG context...")
        full_response_2 = ""
        response_stream_2 = llm.generate_response_stream(history=test_history_2, retrieved_docs=rag_docs, **gen_kwargs)
        for chunk in response_stream_2:
            print(chunk, end="", flush=True)
            full_response_2 += chunk
        print("\n--- End Test Case 2 ---")


        # --- Test Case 3: Conversation with Emotion Instruction ---
        print("\n--- Test Case 3: Conversation with Emotion Instruction ---")
        test_history_3 = [
             {"role": "user", "content": "My program keeps crashing."} # Last user message
        ]
        emotion = "Respond empathetically and offer basic troubleshooting advice."
        print(f"Streaming response with emotion: '{emotion}'...")
        full_response_3 = ""
        response_stream_3 = llm.generate_response_stream(history=test_history_3, emotion_instruction=emotion, **gen_kwargs)
        for chunk in response_stream_3:
            print(chunk, end="", flush=True)
            full_response_3 += chunk
        print("\n--- End Test Case 3 ---")

    except Exception as e:
        logger.exception(f"An error occurred during the LLM example run: {e}")

    print("\nLLM module example finished.")

# Need these imports for the example usage
# import sys # Removed from here
# from pathlib import Path # Removed from here 