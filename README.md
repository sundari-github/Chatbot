# PDF Chatbot

A simple Streamlit-based PDF chatbot that extracts text from uploaded PDF files, converts it into embeddings using OpenAI, and answers user questions using LangChain and FAISS vector search.

## Features

- Upload a PDF file via the Streamlit sidebar
- Extract text from the PDF using `pdfplumber`
- Split the text into searchable chunks
- Create embeddings with `OpenAIEmbeddings`
- Store and search embeddings with `FAISS`
- Answer user questions based on the PDF content only

## Requirements

- Python 3.11+ recommended
- `requirements.txt` includes the main dependencies

## Setup

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Create or update `param.env` with your OpenAI API key:

```text
OPEN_API_KEY=your_openai_api_key_here
```

4. Run the app:

```powershell
streamlit run rag_chatbot.py
```

## Usage

- Open the Streamlit app in your browser
- Upload a PDF file
- Enter a question about the PDF content
- Read the chatbot's answer based only on the uploaded document

## Notes

- The app uses OpenAI embeddings and a local FAISS vector store for retrieval.
- Ensure your `OPEN_API_KEY` is valid and stored in `param.env`.

## Project Files

- `rag_chatbot.py` — main Streamlit application
- `param.env` — API key configuration file
- `requirements.txt` — Python dependencies
- `docs/` — supporting project documentation and sample files