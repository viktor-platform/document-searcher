import os

from openai import OpenAI

from app.AI_search.config import COMPLETIONS_MODEL
from app.AI_search.config import EMBEDDINGS_MODEL
from app.AI_search.config import TEMPERATURE


def get_API_key() -> tuple[str, str]:
    """Get API key and endpoint from app secret"""
    try:
        API_KEY = os.environ["API_KEY"]
        ENDPOINT = os.environ["ENDPOINT"]
    except KeyError:
        from API_KEY import API_KEY  # pylint: disable=import-outside-toplevel
        from API_KEY import ENDPOINT  # pylint: disable=import-outside-toplevel
    return API_KEY, ENDPOINT


def get_chat_completion_gpt(client: OpenAI, questions_and_answers):
    completion = client.chat.completions.create(
        model=COMPLETIONS_MODEL, messages=questions_and_answers, temperature=TEMPERATURE
    )
    return completion


def get_response_message(completion):
    """Gets response message from API call to AzureAI"""
    return completion.choices[0].message.content


def get_embedding(client: OpenAI, text_to_embed: str):
    embedding = client.embeddings.create(input=text_to_embed, model=EMBEDDINGS_MODEL).data[0].embedding
    return embedding
