<div align="center">

# Psych-LLM: Advanced RAG Pipeline for Psychology Research

[![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/yash4agr/casml-psych-llm)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

## 📖 Overview
A Retrieval-Augmented Generation (RAG) system designed for psychology research documentation, developed for the CASML Generative AI Hackathon. This solution combines advanced document retrieval with generative AI to provide context-aware answers from academic literature (OpenStax Psychology (2e), CC BY 4.0).

Key Features:
- PDF document parsing with page-aware chunking
- Hybrid vector database integration (ChromaDB)
- Metadata-enriched text embeddings
- LLM-powered response generation with source attribution

## 🛠️ Technical Stack
- **Embeddings**: TogetherAI/m2-bert-80M-2k-retrieval
- **Vector DB**: ChromaDB
- **LLM**: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
- **Text Processing**: pdfplumber, custom chunking pipeline

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- Together API key

### Setup Instructions

1. **Clone repository**
```bash
git clone https://github.com/yourusername/psych-llm.git
cd psych-llm
```
2. Create a virtual environment
```bash
python -m venv venv
```
3. Activate the virtual environment
```bash 
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

4. Install dependencies
```bash 
pip install -r requirements.txt
```

5. Register and obtain your API key
    - Visit [Together API Key Settings](https://api.together.xyz/settings/api-keys)
    - Generate an API key

6. Export your Together API Key
```bash 
export TOGETHER_API_KEY=<your_api_key>  # On macOS/Linux
set TOGETHER_API_KEY=<your_api_key>  # On Windows
```

7. Run the application
```bash 
python main.py
```