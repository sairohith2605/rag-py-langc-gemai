import os

from langchain_core.prompts import PromptTemplate
from langchain_docling import DoclingLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus


class Extractor:
    VECTOR_DB_COLLECTION_NAME = "doc_embeddings"
    MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI")

    data_transform_prompt_template = PromptTemplate.from_template(
        """
        You will be provided some context, and a question along with that. Generate a human understandable
        response based on the question and the context. The response must answer the question as clearly as possible,
        and you may add good explanation to that as needed. If you cannot answer the question, simply respond with
        "I'm unable to answer the presented question - perhaps I may not have the relevant information". You may generate
        the response in markdown if applicable for cleaner response.
        
        Context: {context}
        Question: {question}
        """
    )

    def __init__(self):
        self.llm_model = self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash"
        )
        self.embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        self.lc_milvus_v_store = Milvus(
            embedding_function=self.embed_model,
            collection_name=self.VECTOR_DB_COLLECTION_NAME,
            primary_field="id",
            vector_field="data_vector",
            connection_args={"URI": self.MILVUS_CONNECTION_URI},
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"},
            enable_dynamic_field=True,
        )

    async def __extract_pdf_embeddings(self, uri: str):
        doc_loader = DoclingLoader(file_path=uri)
        pdf_load = await doc_loader.aload()
        vector_ids = await self.lc_milvus_v_store.aadd_documents(documents=pdf_load)
        return vector_ids

    async def __query_vstore(self, query):
        result_docs = self.lc_milvus_v_store.similarity_search(query=query, k=10)
        return result_docs

    async def process_pdf_from_url(self, uri: str):
        await self.__extract_pdf_embeddings(uri)

    async def generate_llm_response(self, query):
        context_docs = await self.__query_vstore(query=query)
        docs_content = str.join('\n', [x.page_content for x in context_docs])
        prompt = await self.data_transform_prompt_template.ainvoke({
            "context": docs_content,
            "question": query,
        })
        llm_response = await self.llm_model.ainvoke(input=prompt)
        return llm_response.text()
