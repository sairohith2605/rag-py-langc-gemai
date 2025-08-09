# RAG for PDFs with LangChain & Gemini AI ![Static Badge](https://img.shields.io/badge/status-WIP-blue)

### A simple FastAPI app implementing a RAG workflow with Gemini AI models via LangChain

<div align="center">
  <img src="docs/simple_rag_lc_arch.png" width="600px" alt="Architecture Diagram"/>
</div>

#### Overview
The application accepts a PDF file source, and answers queries (in a natural language) about the contents in the PDF. It integrates with Gemini AI with LangChain's interfaces to Google's models.
It uses the enlisted models:
- `gemini-2.0-flash` - For GenAI chat LLM to generate augmented responses with a natural tone
- `text-embedding-004` - To generate the embedding vectors for the PDF and also the query text

> Note - 
> The model can be improved both in terms of the vector database optimization, and a better integration of LangChain's capabilities. That's part of the progress as I learn more.

#### Environment Variables

- `GOOGLE_API_KEY` - The API key for Gemini AI models
- `MILVUS_CONNECTION_URI` - The URL to the Milvus instance
- `MILVUS_USERNAME` - The username to authenticate to Milvus (can be left blank if not applicable)
- `MILVUS_PASSWORD` - The password to authenticate to Milvus (can be left blank if not applicable)
- `MILVUS_TOKEN` - Token to the Milvus cluster (can be left blank if not applicable)
- `EMBED_MODEL` - The model to be used for embedding (currently supported values are `gemini` (default) and `qwen`)
- `LLM_MODEL` - The LLM model to be used for chat (currently supported values are `gemini` (default) and `qwen`)
- `QWEN_EMBEDDINGS_URI` - The URL to the Qwen Embedding model
- `QWEN_LLM_URI` - The URL to the Qwen Chat model