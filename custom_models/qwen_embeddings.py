import os
import requests

from langchain_core.embeddings import Embeddings


class QwenEmbeddings(Embeddings):
    """`Qwen Text Embeddings`.

        Add the Qwen Model's URL in the environment as ``QWEN_EMBEDDINGS_URI`` for embedding generations.
        """
    def __init__(self):
        self.qwen_uri = os.getenv("QWEN_EMBEDDINGS_URI")

    def __send_requests(self, text: list[str]):
        print('Processing embedding requests with Qwen')
        response = requests.post(self.qwen_uri, json={"inputs": text})
        response.raise_for_status()
        if isinstance(response.json(), list):
            print('Embeddings processed successfully. Size of embeds: ', len(response.json()))
            return response.json()
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.__send_requests(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.__send_requests(text=[text])[0]
