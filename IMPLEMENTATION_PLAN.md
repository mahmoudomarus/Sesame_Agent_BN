# Implementation Plan: Low-Latency Voice Agent (Local + RAG + GUI)

## Goal

Create a low-latency, high-quality voice agent using Sesame CSM 1B for TTS, running locally on Linux (RTX 4090). The agent should:
- Listen via microphone.
- Use VAD (Silero) for endpointing.
- Transcribe speech (faster-whisper).
- **(New)** Query a local knowledge base using RAG.
- Generate conversational responses using an LLM (Phi-3 Medium), incorporating RAG context or providing entertainment if no context is found.
- Synthesize responses using Sesame CSM 1B (via `SesameAILabs/csm` code).
- Play back audio.
- Handle interruptions gracefully.
- **(New)** Offer a GUI for control (pause, interrupt), visualization, and potentially tuning (e.g., emotional expression).
- Aim for an experience comparable to or better than commercial offerings, prioritizing low perceived latency despite potential TTS limitations.

## Core Technologies

- **OS:** Linux
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **Core:** Python 3.10+
- **Environment:** `venv`
- **STT:** `faster-whisper` (`large-v3`)
- **VAD:** `silero-vad`
- **LLM Engine:** `vLLM`
- **LLM Model:** `microsoft/Phi-3-medium-128k-instruct` (or similar fitting VRAM)
- **TTS:** `sesame/csm-1b` (via `SesameAILabs/csm` repository code) + `meta-llama/Llama-3.2-1B` (required by CSM)
- **Audio:** `sounddevice`, `numpy`
- **RAG:** `langchain` or `llama-index`, Vector Store (e.g., `faiss-gpu`, `chromadb`)
- **GUI:** TBD (e.g., `PyQt`, `tkinter`, `customtkinter`, `streamlit`)
- **Config:** `PyYAML`, `python-dotenv`
- **Hugging Face:** `huggingface-hub`, `huggingface-cli`

## Phases

### Phase A: Linux Environment Setup & Verification

1.  **System Prerequisites:**
    *   Verify `nvidia-smi` shows RTX 4090 and compatible CUDA Toolkit. Install/update if needed.
    *   Install `build-essential`, `cmake`, `ffmpeg`, `python3.10+`, `python3-pip`, `python3.10-venv`.
    *   Navigate to project directory.
    *   Create/Activate venv: `python3 -m venv venv && source venv/bin/activate`.
    *   **Login to Hugging Face:** Run `huggingface-cli login` and authenticate. This is required for downloading CSM and Llama 3.2. **Ensure your HF account has requested/been granted access to `sesame/csm-1b` and `meta-llama/Llama-3.2-1B`.**

### Phase B: Dependency Installation

1.  **Clone Sesame CSM Repository:**
    ```bash
    mkdir external && cd external
    git clone https://github.com/SesameAILabs/csm.git
    cd ..
    # Add external/csm to .gitignore
    ```
2.  **Prepare Merged `requirements.txt`:**
    *   Create/update the main `requirements.txt` in the project root.
    *   **Include:**
        *   Core libs (`torch` matching CUDA, `torchaudio`, `sounddevice`, `numpy`, `pyyaml`, `python-dotenv`, `tqdm`). **Ensure torch version matches `csm`'s requirements if specified.**
        *   STT (`faster-whisper-ctranslate2`).
        *   VAD (`silero-vad`). Download handled by torch.hub later.
        *   LLM Engine (`vllm`).
        *   RAG libs (`langchain`, `langchain-community`, `langchain-huggingface` or `llama-index`, `faiss-gpu` or `chromadb`, PDF/document loaders like `pypdf`).
        *   Hugging Face (`huggingface-hub`).
    *   **Merge/Verify:** Check `external/csm/requirements.txt`. Add any *missing* essential dependencies from there to the main `requirements.txt`. Prioritize CSM's versions if conflicts arise, especially for `torch`, `transformers`.
3.  **Install Requirements:**
    *   **(Optional but Recommended):** Install `csm` dependencies first to catch conflicts early: `pip install -r external/csm/requirements.txt`.
    *   Install main requirements: `pip install -r requirements.txt`.

### Phase C: Model Acquisition & Configuration

1.  **Download Models:** Create `models/` directory if it doesn't exist.
    *   **STT:** `faster-whisper-large-v3` (via HF Hub or manually) -> `models/faster-whisper-large-v3`
    *   **LLM:** `microsoft/Phi-3-medium-128k-instruct` (via `huggingface-cli download ... --local-dir ...` or git lfs) -> `models/Phi-3-medium-128k-instruct`
    *   **CSM TTS:** `sesame/csm-1b` (via HF Hub) -> `models/sesame-csm-1b`
    *   **CSM Dep:** `meta-llama/Llama-3.2-1B` (via HF Hub) -> `models/Llama-3.2-1B`
2.  **Configure Paths & Secrets:**
    *   **Create `.env`:** From `.env.example`, add your `HUGGING_FACE_TOKEN`.
    *   **Edit `config/config.yaml`:**
        *   Update all `model_path` entries to reflect the actual local download paths.
        *   Verify `llm.engine: vllm`.
        *   Adjust `llm.engine_kwargs.gpu_memory_utilization` (start ~0.8).
        *   Add RAG config section (knowledge base path, vector store path, chunk size, etc.).
        *   Add placeholder for GUI settings if needed.

### Phase D: Core Module Implementation

*(Implement `TODO`s and new logic)*

1.  **TTS Module (`src/tts_module.py`):**
    *   **Major Overhaul:** Remove previous streaming logic.
    *   Import `load_csm_1b`, `Segment` from `external.csm.generator` (adjust path if needed).
    *   `__init__`: Load the model using `load_csm_1b(device=device)`. Store the returned `generator` object. Get and store `generator.sample_rate`.
    *   `synthesize_speech(text: str, context: list = []) -> Tuple[np.ndarray | None, int]`:
        *   Call `self.generator.generate(text=text, speaker=0, context=context, max_audio_length_ms=...)`. Speaker ID might need configuration.
        *   **Handle Context:** Prepare the `context` list using `Segment` objects if voice prompting data is provided (initially pass `[]`).
        *   **NO STREAMING:** This call is expected to be **blocking**.
        *   Return the resulting audio tensor (convert to float32 numpy if needed) and sample rate.
    *   Remove `synthesize_speech_stream`.
2.  **LLM Module (`src/llm_module.py`):**
    *   Keep `generate_response_stream` (LLM streaming is still beneficial).
    *   Implement actual vLLM streaming logic in `generate_response_stream`.
    *   Refine `_format_input_chatml` for Phi-3, incorporating placeholders for RAG context.
    *   **RAG Integration Point:** The function calling the LLM stream will need to potentially insert retrieved context into the `history` or `prompt` passed to the stream generator, using the chosen formatting.
    *   **Emotional Control:** Modify prompt formatting to include instructions based on desired emotion (e.g., "Respond humorously:", "Respond empathetically:").
3.  **STT Module (`src/stt_module.py`):**
    *   Implement `faster-whisper` loading and transcription as planned previously.
4.  **RAG Module (`src/rag_module.py`) (New):**
    *   Create the file.
    *   Choose framework (`Langchain` or `LlamaIndex`).
    *   Implement `RAGManager` class:
        *   `__init__`: Load config (KB path, vector store path). Initialize document loaders, text splitter, embedding model (e.g., local sentence-transformer), vector store (`faiss-gpu` recommended for speed).
        *   `build_index(docs_path)`: Load docs, split, embed, build/save FAISS index. Run this offline initially.
        *   `retrieve(query: str, top_k: int) -> List[str]`: Load index, perform similarity search, return relevant document chunks.
5.  **Knowledge Base:**
    *   Create a `knowledge_base/` directory.
    *   Place sample `.txt`, `.pdf`, or other documents inside for testing RAG.

### Phase E: Orchestration & GUI Implementation (`main.py`, `gui.py`)

1.  **`main.py` Updates:**
    *   Load `.env` using `python-dotenv`.
    *   Initialize `RAGManager`.
    *   **Modify `run_inference_pipeline`:**
        *   After STT, call `rag_manager.retrieve(user_text)`.
        *   Prepare LLM input: Format prompt/history including retrieved RAG context if found. Add emotional instruction if provided (from GUI/config).
        *   Call `llm_model.generate_response_stream`.
        *   **Consume LLM Stream:** Iterate through the stream, accumulating the full response text (`agent_response_text`).
        *   **(New TTS Call):** Once LLM stream is *finished*, call the **blocking** `tts_model.synthesize_speech(agent_response_text)`.
        *   **(Playback):** Put the *entire* synthesized audio chunk onto the `audio_playback_queue`. The playback worker remains non-blocking.
        *   Update history with the fully captured `agent_response_text`.
    *   VAD integration: Implement Silero VAD logic.
    *   Interruption: Logic needs to stop playback (clear queue) and potentially signal the LLM stream to stop (best effort).
2.  **`gui.py` (New):**
    *   Choose GUI framework (`customtkinter` is often good for modern look/feel with Python).
    *   Create the main GUI window class.
    *   **Components:**
        *   Start/Stop button for the agent.
        *   Status display (Listening, Thinking, Speaking, VAD detected).
        *   Conversation log display area.
        *   Manual Interrupt button.
        *   (Optional) Pause/Resume button.
        *   **(New)** Sliders or controls for emotional parameters (e.g., Humor: 0-1). These values need to be passed to `main.py` to influence the LLM prompt.
        *   (Optional) Microphone input level visualizer.
    *   **Integration:** The GUI needs to communicate with the main agent logic running in `main.py`. This often involves using thread-safe queues or callbacks to:
        *   Start/stop the agent threads in `main.py`.
        *   Send interrupt/pause signals.
        *   Receive status updates and conversation text from `main.py` to display.
        *   Send emotional parameter values to `main.py`.

### Phase F: Testing, VRAM Management & Tuning (Iterative)

1.  **Build RAG Index:** Run a separate script or add logic to `rag_module.py` to build the initial vector index from documents in `knowledge_base/`.
2.  **Initial Run:** `python gui.py` (assuming it starts the main agent logic) or `python main.py`. Debug startup errors.
3.  **VRAM Check:** Monitor `nvidia-smi`. CSM + Llama 3.2 + Phi-3 Medium + Whisper Large + VLLM + FAISS-GPU will be demanding. **The 24GB 4090 *might* be sufficient, but be prepared to optimize:**
    *   Reduce `gpu_memory_utilization`.
    *   Use STT/LLM quantization (`int8`, AWQ, etc.). Check vLLM/faster-whisper docs.
    *   Unload RAG index between queries if needed (trade-off speed vs memory).
    *   Switch to smaller models (Phi-3-mini, Whisper-medium/small).
4.  **Functional Testing:** Test conversation flow, RAG retrieval (ask questions about KB docs), fallback behavior, GUI controls (interrupt, sliders).
5.  **Latency Perception:** How long after you stop speaking does the agent *start* responding? This will be impacted by the non-streaming TTS. Is the delay acceptable?
6.  **Tuning:** Adjust VAD, LLM generation params, RAG retrieval params (`top_k`), emotional prompt injection.
7.  **(Future) Voice Cloning:** Implement logic to load user-provided audio (`utterance_X.wav`), create `Segment` objects, and pass them to `tts_model.synthesize_speech(..., context=segments)` to test voice prompting.

## Key Challenges & Notes

-   **TTS Latency:** The lack of documented streaming in `SesameAILabs/csm` is the biggest hurdle for achieving ultra-low latency comparable to fully streaming solutions. The agent's response will only start playing *after* the entire text is generated by the LLM *and* the entire audio is synthesized by CSM.
-   **VRAM Management:** Fitting all models (STT, LLM, CSM TTS, CSM Llama Dep, RAG Embeddings, FAISS index) onto 24GB requires careful monitoring and likely optimization/quantization.
-   **GUI Integration:** Building a responsive GUI that communicates effectively with the multi-threaded backend requires careful design (queues, signals/slots).
-   **RAG Quality:** Retrieval accuracy depends on chunking strategy, embedding model, and document quality.
-   **Emotional Control Nuance:** Simple sliders might be coarse. Achieving nuanced emotional expression depends heavily on the LLM's ability to follow prompt instructions and potentially on TTS features (if CSM offers style control beyond context).
-   **CSM API Details:** The exact methods and parameters for `SesameAILabs/csm` need verification by inspecting the cloned code.

This plan provides a roadmap for building the enhanced local voice agent. 