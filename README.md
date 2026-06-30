# Conversational AI with Retrieval-Augmented Generation (RAG) and Agentic Tool Integration

An advanced, hybrid Natural Language Processing (NLP) framework designed to bridge the gap between generative context modeling and deterministic execution environments. 

**Developer:** Muhammad Subhan Shahid 
**Program:** BS Artificial Intelligence at FAST-NU

---

## Hugging Face Artifacts
To maintain a lightweight and professional codebase, the heavy neural network model configurations, fine-tuned weight parameters, and binary dense vector search trees are hosted separately on Hugging Face:

* **Fine-Tuned Model & FAISS Binaries:** [https://huggingface.co/subhan1501/DialoGPT-Agentic-RAG-Controller](https://huggingface.co/subhan1501/DialoGPT-Agentic-RAG-Controller)
* **Knowledge Base & Dialogue Data:** [https://huggingface.co/subhan1501/DailyDialog-Agentic-KnowledgeBase](https://huggingface.co/subhan1501/DailyDialog-Agentic-KnowledgeBase)

*(Note: The application is configured to automatically download these weights via the `huggingface_hub` library upon first execution).*

---

## Core System Architecture
The application runs as a unified cognitive agent executing logic pathways handled dynamically by a custom **Semantic Intent Router**:

1. **Agentic Layer (Tools):** Monitors explicit keywords to execute deterministic Python calculations, system time metrics, or external mocked utilities.
2. **RAG Layer (Vector Memory):** Converts text inputs into dense mathematical vectors using `all-MiniLM-L6-v2`. It queries a local **FAISS Vector Database** to find matching context, injecting relevant factual strings if the calculated L2 space distance passes the threshold of `< 1.5`.
3. **Generative Layer (Chatterbox Fallback):** Open-domain prompts bypass search layers entirely, routing automatically to an autoregressive **DialoGPT-small** model fine-tuned extensively on multi-turn dialogue structures.

---

## Performance Matrix & Evaluation
The generative chat sequences were rigorously tested against structured gold-standard human dialogue arrays using standard NLP evaluation architectures:

* **Average BLEU Score:** `0.4120` (Measures exact n-gram overlap)
* **Average ROUGE-1 F1:** `0.6515` (Measures keyword recall and precision)
* **Average ROUGE-L F1:** `0.6280` (Measures longest common subsequence for flow)

---

## Repository Structure

```text
conversational_agent/
│
├── app/
│   ├── agent.py          # Core Cognitive Controller & Semantic Intent Router
│   ├── tools.py          # Deterministic Python scripts (Math, Time, APIs)
│   └── ui.py             # User Dashboard constructed via Streamlit
│
├── notebooks/
│   ├── 00-preprocess-data.ipynb
│   ├── 01_baseline_tfidf.ipynb
│   └── 02_train_model.ipynb
│
├── results/
│   ├── eda_word_counts.png
│   └── evaluation_metrics_report.txt
│
├── scripts/
│   ├── build_vector_db.py
│   ├── evaluate_metrics.py
│   └── generate_eda.py
│
├── test_chat.py          # Terminal-based evaluation harness
├── requirements.txt      # System dependencies
└── README.md             # Project documentation
```
## Quick Setup & Interface Initiation
### Prerequisites
Python 3.8+

Git

### Installation Steps
1. Clone the Codebase:

```Bash
git clone [https://github.com/subhans1501/Conversational-Agentic-RAG-System.git](https://github.com/subhans1501/Conversational-Agentic-RAG-System.git)
cd Conversational-Agentic-RAG-System
```
2. Set Up Environments:

```Bash
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Linux/Mac:
source venv/bin/activate
```
3. Install Dependencies:

```Bash
pip install -r requirements.txt
```
4. Boot Up the Web Interface Dashboard:

```Bash
streamlit run app/ui.py
```
The Streamlit server will initialize, automatically connect to the Hugging Face hub to cache the required FAISS binaries, and launch the interactive UI in your default web browser.

## Limitations & Future Roadmap
* **Hardware Footprints:** Generative runtime inference defaults to local CPU thread execution, introducing slight structural latency over deep sequences compared to GPU acceleration.

* **API Fallbacks:** The current weather tool utilizes a mocked internal dictionary to prevent rate-limiting. Future iterations will integrate live web-search (e.g., SerpApi) for open-world grounding.

* **Model Scaling:** Planned future iterations aim to migrate the core generative pipeline from DialoGPT to deeper reasoning topologies, such as Llama-3 or Mistral.
