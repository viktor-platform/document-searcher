"""Copyright (c) 2023 VIKTOR B.V.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.
VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List

from openai import OpenAI
from scipy import spatial
from viktor import progress_message

from .helper_functions import get_embedding


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


def get_question_for_language(current_question: str) -> list[dict]:
    """Retrieves the language from the question and returns it as an instruction."""
    return [
        {
            "role": "user",
            "content": f"What language is the following text? Answer in a single word "
            f"{current_question}. Provide your answer as an instruction, "
            f"such as: 'Answer in Dutch', 'Answer in English'",
        }
    ]


def get_question_with_context(current_question: dict, context: str):
    """First create embedding of the question. Then, create the context. Return answer as conversation."""
    prompt = f"""
             Question: {current_question["content"]}  \n
             Answer the question based on the context below. When you don't know the answer, say "I don't know the 
             answer, based on the provided context.  \n
             Context: {context}\n
             """
    question_with_context = {"role": "user", "content": prompt}
    return question_with_context


def create_context(client: OpenAI, current_question: str, df, context_number):
    """Create a context for a question by finding the most similar context from the dataframe"""

    progress_message("Creating context for question")
    question_embedded = get_embedding(client, current_question)

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
