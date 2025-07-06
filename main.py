from dotenv import load_dotenv
from fastapi import FastAPI

from extractor import Extractor
from models.document_uri_request import DocumentURIRequest
from models.query_request import QueryRequest

app = FastAPI()
load_dotenv()
extractor = Extractor()

@app.post("/document/uri")
async def process_document_from_uri(document_uri: DocumentURIRequest):
    await extractor.process_pdf_from_url(document_uri.uri)
    return {"result": "Document has been processed successfully"}


@app.post("/query")
async def process_query(query: QueryRequest):
    query_result = await extractor.generate_llm_response(query.query)
    return {"result": query_result}
