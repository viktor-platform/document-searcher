import random
from typing import List

import numpy as np
import openai
from openai import InvalidRequestError
from openai.error import RateLimitError, APIError, AuthenticationError, ServiceUnavailableError
from scipy import spatial
from viktor import progress_message, UserError

from config import EMBEDDINGS_MODEL, RETRIES, COMPLETIONS_MODEL, TEMPERATURE, RETRY_MESSAGE


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]
    return distances


def get_question_for_language(current_question):
    """Retrieves the language from the question and returns it as an instruction."""
    return [{"role": "user", "content": f"What language is the following text? Answer in a single word "
                                        f"{current_question}. Provide your answer as an instruction, "
                                        f"such as: 'Answer in Dutch', 'Answer in English'"}]


def get_question_with_context(current_question, context):
    """First create embedding of the question. Then, create the context. Return answer as conversation."""
    prompt = f"""
             Question: {current_question["content"]}  \n
             Answer the question based on the context below. When you don't know the answer, say "I don't know the 
             answer, based on the provided context.  \n
             Context: {context}\n
             """
    question_with_context = {"role": "user", "content": prompt}
    return question_with_context


def create_context(current_question, df, context_number):
    """Create a context for a question by finding the most similar context from the dataframe"""

    progress_message("Creating context for question")
    question_embedded = openai.Embedding.create(input=current_question, engine=EMBEDDINGS_MODEL)["data"][0]["embedding"]

    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(question_embedded, df["embeddings"].values, distance_metric="cosine")
    df = df.sort_values("distances", ascending=True).reset_index()

    # Sort by distance and add the text to the context until the context is too long
    context_list = []
    metadata_list = []
    for i, row in df.iloc[:context_number].iterrows():
        context_list.append(row["text"])
        metadata_list.append({"page_number": row["page_number"], "source": row["source"]})
    context_list = [row["text"] for i, row in df.iloc[:context_number].iterrows()]

    # Return the context
    context = "\n\n###\n\n".join(context_list)
    return context, metadata_list, context_list


def get_response_message(completion):
    return completion["choices"][0]["message"]["content"]


def get_chat_completion_gpt(questions_and_answers):
    for idx in range(RETRIES):
        try:
            completion = openai.ChatCompletion.create(
                model=COMPLETIONS_MODEL, messages=questions_and_answers, temperature=TEMPERATURE
            )
            break
        except InvalidRequestError:
            raise UserError(
                "InvalidRequestError: The number of maximum tokens [4097] is exceeded. Please start a new "
                "conversation and try to get to the desired answer in less requests."
            )
        except RateLimitError:
            if idx < RETRIES - 1:
                progress_message(random.choice(RETRY_MESSAGE))
                continue
            raise UserError("RateLimitError: The ChatGPT model is currently overloaded. Please retry your request.")
        except APIError:
            if idx < RETRIES - 1:
                progress_message(random.choice(RETRY_MESSAGE))
                continue
            raise UserError(
                "API Error: The server had an error while processing your request. Sorry about that! "
                "Please retry your request."
            )
        except AuthenticationError:
            raise UserError(
                "AuthenticationError: There seems to be some problems with the current API key. Please report the "
                "error and we will try to fix this as soon as possible."
            )
        except ServiceUnavailableError:
            if idx < RETRIES - 1:
                progress_message(random.choice(RETRY_MESSAGE))
                continue
            raise UserError("ServiceUnavailableError: The server is overloaded or not ready yet.")
    return completion
