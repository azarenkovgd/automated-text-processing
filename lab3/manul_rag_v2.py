import json

import ollama
import faiss
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from common.start_logging import start_logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def generate_embedding(text: str) -> Optional[np.ndarray]:
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return np.array(response['embedding']).astype('float32')
    except Exception as e:
        logging.error(f"Error generating embedding for text '{text}': {e}")
        return None


def create_faiss_index(data: List[Dict[str, str]]) -> faiss.IndexFlatL2:
    logging.info("Creating FAISS index...")
    embeddings: List[np.ndarray] = []

    for entry in data:
        embedding = generate_embedding(str(entry))
        if embedding is not None:
            embeddings.append(embedding)
        else:
            raise Exception("Failed to generate embedding for an entry.")

    if not embeddings:
        raise Exception("No embeddings generated.")

    dimension: int = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    logging.info("FAISS index created successfully.")
    return index


def search_in_dataset(
        query: str,
        index: faiss.IndexFlatL2,
        data: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    logging.info(f"Searching for query: '{query}'")
    query_embedding = generate_embedding(query)

    if query_embedding is None:
        raise Exception("Failed to generate query embedding.")

    k = 5
    distances, indices = index.search(np.array([query_embedding]), k)
    results = []
    for i, idx in enumerate(indices[0]):
        result = data[idx]
        result_distance = distances[0][i]
        results.append({"data": result, "distance": result_distance})

    logging.info(f"Search completed. Top {k} matches: {results}")
    return results


def send_question_to_model(query: str) -> str:
    logging.info("Sending question to model...")

    response = ollama.chat(model='llama3.1:latest', messages=[
        {
            "role": "system", "content": '''
            You are assistant that helps answering questions.
            '''
        },
        {
            "role": "user", "content": query
        }])
    answer = response["message"]["content"]
    return answer


def send_question_to_model_with_context(query: str, top_entries: List[Dict[str, Any]]) -> str:
    context_lines = "\n".join([f"piece{i + 1}: {entry['data']}" for i, entry in enumerate(top_entries)])

    logging.info("Sending question to model with context...")
    response = ollama.chat(model='llama3.1:latest', messages=[
        {
            "role": "system", "content": '''
            You are assistant that helps answering questions about dataset. You are given some entries in the dataset, 
            that can help you answer.
            Include ALL possible results based on tool input. But it is not necessary, if some of them are not 
            relevant.
            '''
        },
        {
            "role": "tool", "content": context_lines
        },
        {
            "role": "user", "content": query
        }])
    answer = response["message"]["content"]
    return answer


def print_response(user_query: str, dataset: list[dict]) -> None:
    faiss_index = create_faiss_index(dataset)

    top_matches = search_in_dataset(user_query, faiss_index, dataset)

    response = send_question_to_model_with_context(user_query, top_matches)

    logging.info(f"Model answered: {response}")

    response_without_context = send_question_to_model(user_query)

    logging.info(f"Model answered without context: {response_without_context}")


def main():
    start_logging()
    # with open('data/people.json', 'r') as file:
    #     dataset = json.load(file)
    #
    # main("Who has a profession associated with money?", dataset)

    with open('../lab2/data/documentation.json', 'r', encoding="utf-8") as file:
        dataset = json.load(file)

    print_response("How to create async handler of some command, using python-telegram-bot", dataset)


if __name__ == '__main__':
    main()
