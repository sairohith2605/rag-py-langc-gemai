# RAG for PDFs with LangChain & Gemini AI ![Static Badge](https://img.shields.io/badge/status-WIP-blue)

### A simple FastAPI app implementing a RAG workflow with Gemini AI models via LangChain

![Simple RAG Model](docs/simple_rag_lc_arch.png)

#### Overview
The application accepts a PDF file source, and answers queries (in a natural language) about the contents in the PDF. It integrates with Gemini AI with LangChain's interfaces to Google's models.
It uses the enlisted models:
- `gemini-2.0-flash` - For GenAI chat LLM to generate augmented responses with a natural tone
- `text-embedding-004` - To generate the embedding vectors for the PDF and also the query text
