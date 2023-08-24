import os

import openai
from viktor.core import progress_message

from config import SYSTEM_MESSAGE, N_CONTEXT
from context import create_context, get_question_with_context, get_chat_completion_gpt, get_response_message, \
    get_question_for_language


def get_API_key():
    try:
        API_KEY, _, _, _ = os.environ["VIKTOR_APP_SECRET"].split("|")
    except KeyError:
        from API_KEY import API_KEY  # pylint: disable=import-outside-toplevel
    return API_KEY


class RetrievalAssistant:
    """Class for constructing the conversation and making the API calls to OpenAI"""

    def __init__(self, question, df):
        self.question = question
        self.context = ""
        self.metadata_list = []
        self.context_list = []
        self.current_question = {}
        self.df = df
        API_KEY = get_API_key()

        openai.api_key = API_KEY

        self._set_current_question(self.question)
        self._create_context()

    def _create_context(self):
        self.context, self.metadata_list, self.context_list = create_context(self.question, self.df, N_CONTEXT)

    def _set_current_question(self, question):
        self.current_question = {"role": "user", "content": question}

    def ask_assistant(self):
        progress_message("Setting up question...")
        completion_question = get_chat_completion_gpt(get_question_for_language(self.current_question))
        language_question = get_response_message(completion_question)

        questions_and_answers = []
        questions_and_answers.append(get_question_with_context(self.current_question, self.context))

        # Insert at the end to make it more accurate
        questions_and_answers.append({"role": "system", "content": SYSTEM_MESSAGE + language_question})
        progress_message("Prompt is sent to OpenAI, waiting for response...")
        completion = get_chat_completion_gpt(questions_and_answers)
        progress_message("Answer received from ChatGPT, saving results...")
        response_message = get_response_message(completion)
        return response_message
