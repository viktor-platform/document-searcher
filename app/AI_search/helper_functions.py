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

import os

from openai import OpenAI

from app.AI_search.config import COMPLETIONS_MODEL
from app.AI_search.config import EMBEDDINGS_MODEL
from app.AI_search.config import TEMPERATURE


def get_API_key() -> tuple[str, str, str]:
    """Get API key and endpoint from app secret"""
    try:
        API_KEY = os.environ["API_KEY"]
        ENDPOINT = os.environ["ENDPOINT"]
        API_VERSION = os.environ["API_VERSION"]
    except KeyError:
        from API_KEY import API_KEY  # pylint: disable=import-outside-toplevel
        from API_KEY import ENDPOINT  # pylint: disable=import-outside-toplevel
        from API_KEY import API_VERSION  # pylint: disable=import-outside-toplevel
    return API_KEY, ENDPOINT, API_VERSION


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
