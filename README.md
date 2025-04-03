# Bignoodle Voice Agent

This project implements a low-latency, locally-run voice agent using advanced AI models for speech-to-text, language understanding, RAG, and text-to-speech.

## Goal

Create a high-quality voice assistant that:
- Runs locally on Linux with NVIDIA GPU (RTX 4090 tested).
- Listens via microphone with VAD (Silero).
- Transcribes speech accurately (faster-whisper large-v3).
- Queries a local knowledge base using RAG (Langchain/FAISS).
- Generates conversational responses (Phi-3 Medium) with RAG context or for entertainment.
- Synthesizes speech using Sesame CSM 1B.
- Offers a GUI for control and visualization (CustomTkinter).
- Handles interruptions.

## Core Technologies

- **OS:** Linux
- **GPU:** NVIDIA GPU (RTX 4090 w/ 24GB VRAM recommended)
- **Core:** Python 3.10+
- **Environment:** `venv`
- **STT:** `faster-whisper` (`large-v3`)
- **VAD:** `silero-vad`
- **LLM Engine:** `vLLM`
- **LLM Model:** `microsoft/Phi-3-medium-128k-instruct`
- **TTS:** `sesame/csm-1b` (via `SesameAILabs/csm` code)
- **TTS Dep:** `meta-llama/Llama-3.2-1B`
- **Audio:** `sounddevice`, `numpy`
- **RAG:** `langchain`, `faiss-gpu`, `sentence-transformers`
- **GUI:** `customtkinter`
- **Config:** `PyYAML`, `python-dotenv`
- **Hugging Face:** `huggingface-hub`

## Setup Instructions

**1. Prerequisites:**

*   Linux environment.
*   NVIDIA GPU with compatible CUDA Toolkit installed. Verify with `nvidia-smi`.
*   Install system dependencies: `sudo apt update && sudo apt install build-essential cmake ffmpeg python3.10 python3-pip python3.10-venv` (adjust package names if needed for your distribution).
*   Python 3.10 or higher.

**2. Clone Repository:**

```bash
git clone https://github.com/mahmoudomarus/BigNoodle_AGENT.git
cd BigNoodle_AGENT
```

**3. Create Virtual Environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**4. Hugging Face Login & Access:**

*   Log in to Hugging Face: `huggingface-cli login` (requires a User Access Token with **read** permissions).
*   **Crucially**, request access on the Hugging Face Hub for the gated models:
    *   [`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b)
    *   [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)
    You **must** be granted access before you can download these models.

**5. Install Dependencies:**

*   Clone the required CSM repository:
    ```bash
    mkdir external && cd external
    git clone https://github.com/SesameAILabs/csm.git
    cd ..
    ```
*   Install Python packages (ensure CUDA version in `requirements.txt` matches your toolkit):
    ```bash
    # Double-check torch/torchaudio lines in requirements.txt for correct CUDA suffix (e.g., cu118, cu121)
    pip install -r requirements.txt
    ```

**6. Download Models:**

*   Create the models directory: `mkdir models`
*   Download using `huggingface-cli` (ensure you are logged in and have access to gated models):
    ```bash
    # STT (Whisper)
huggingface-cli download --repo-type model openai/whisper-large-v3 --local-dir models/faster-whisper-large-v3

    # LLM (Phi-3)
huggingface-cli download --repo-type model microsoft/Phi-3-medium-128k-instruct --local-dir models/Phi-3-medium-128k-instruct

    # TTS (CSM) - Requires granted access
huggingface-cli download --repo-type model sesame/csm-1b --local-dir models/sesame-csm-1b

    # TTS Dependency (Llama) - Requires granted access
huggingface-cli download --repo-type model meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B
    ```

**7. Configure Environment:**

*   Copy the example environment file: `cp .env.example .env`
*   Edit the `.env` file and add your Hugging Face **read** token:
    ```env
    HUGGING_FACE_TOKEN="hf_YOUR_TOKEN_HERE"
    ```

**8. Configure Application (`config/config.yaml`):**

*   Review the `config/config.yaml` file.
*   The default paths should match the model download locations.
*   Adjust `llm.engine_kwargs.gpu_memory_utilization` (e.g., `0.7`) and `llm.engine_kwargs.max_model_len` (e.g., `8192`) based on your GPU memory. Phi-3 Medium can be memory-intensive.
*   Modify audio input/output device indices if needed.
*   Configure RAG paths (`knowledge_base_path`, `index_path`) if defaults are not suitable.

**9. Prepare Knowledge Base & RAG Index:**

*   Create the knowledge base directory (if it doesn't exist): `mkdir knowledge_base`
*   Place your documents (.txt, .pdf) into the `knowledge_base` directory.
*   Build the FAISS index. You can run the RAG module directly for this (it includes example logic to build the index):
    ```bash
    # Ensure venv is active
    python src/rag_module.py
    ```
    This will create an index in `models/faiss_index_example`. For the main application, ensure the index path in `config.yaml` (`models/faiss_index` by default) exists and is built using your actual documents. You might want to create a dedicated script for building the production index.

## Running the Agent

Once setup is complete:

```bash
# Ensure venv is active
source venv/bin/activate

# Run the main application (which launches the GUI)
python main.py
```

Use the GUI controls (Start/Stop, Interrupt) to interact with the agent.

## Notes & Troubleshooting

*   **VRAM:** This application is VRAM-intensive. Loading Whisper Large, Phi-3 Medium, CSM 1B + Llama 3.2 1B, and RAG components requires substantial GPU memory (24GB recommended). If you encounter CUDA Out-of-Memory errors:
    *   Lower `gpu_memory_utilization` in `config/config.yaml`.
    *   Ensure `max_model_len` is set for the LLM in the config.
    *   Consider using smaller models (e.g., `Phi-3-mini`, `whisper-medium/small`).
    *   Try quantization options (e.g., `int8` for `faster-whisper`, check vLLM docs for LLM options).
*   **TTS Latency:** The CSM TTS model used here does not currently support streaming. There will be a noticeable delay between the LLM finishing generation and the audio playback starting, as the entire response must be synthesized first.
*   **Dependencies:** Ensure all dependencies installed correctly, especially `torch` with the correct CUDA version and `faiss-gpu`. # BigNoodle_AGENT
