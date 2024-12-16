import json
import logging
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from common.start_logging import start_logging
from lab2.manual_rag import send_question_to_model


def create_index_from_json(dataset: list[dict]) -> VectorStoreIndex:
    documents = [Document(text=json.dumps(entry)) for entry in dataset]

    embed_model = OllamaEmbedding(model_name='nomic-embed-text')
    llm = Ollama(model='llama3.1:latest')

    Settings.embed_model = embed_model
    Settings.llm = llm

    logging.info("Creating index from JSON data...")
    index = VectorStoreIndex.from_documents(documents)
    logging.info("Index created successfully.")
    return index


def query_index(index: VectorStoreIndex, query: str) -> str:
    logging.info(f"Querying the index with: '{query}'")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return str(response)


def create_query_function(dataset: list[dict]):
    index = create_index_from_json(dataset)

    return lambda query: query_index(index, query)


def main():
    start_logging()

    with open('../lab2/data/documentation.json', 'r', encoding="utf-8") as file:
        dataset = json.load(file)

    query_function = create_query_function(dataset)

    user_query = "List some performance optimizations for python-telegram-bot"

    response = query_function(user_query)

    logging.info(f"Response: {response}")

    response_without_context = send_question_to_model(user_query)

    logging.info(f"Response without context: {response_without_context}")

if __name__ == '__main__':
    main()
