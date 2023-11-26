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

import openai
from openai.lib.azure import AzureOpenAI
from viktor.core import progress_message

from .config import MAX_RETRIES
from .config import N_CONTEXT
from .config import SYSTEM_MESSAGE
from .context import create_context
from .context import get_question_for_language
from .context import get_question_with_context
from .helper_functions import get_API_key
from .helper_functions import get_chat_completion_gpt
from .helper_functions import get_response_message


class RetrievalAssistant:
    """Class for constructing the conversation and making the API calls to AzureAI"""

    def __init__(self, question, df):
        self.question = question
        self.context = ""
        self.metadata_list = []
        self.context_list = []
        self.current_question = {}
        self.df = df
        API_KEY, ENDPOINT = get_API_key()
        self.client = AzureOpenAI(
            api_key=API_KEY, api_version="2023-10-01-preview", azure_endpoint=ENDPOINT, max_retries=MAX_RETRIES
        )
        openai.api_key = API_KEY
        openai.api_base = ENDPOINT
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"

        self._set_current_question(self.question)
        self._create_context()

    def _create_context(self):
        """Set the context for the question"""
        self.context, self.metadata_list, self.context_list = create_context(
            self.client, self.question, self.df, N_CONTEXT
        )

    def _set_current_question(self, question: str):
        """Converts current question to correct format"""
        self.current_question = {"role": "user", "content": question}

    def ask_assistant(self):
        """Method for preparing the question and then asking it to AzureAI"""
        progress_message("Setting up question...")
        completion_question = get_chat_completion_gpt(self.client, get_question_for_language(self.current_question))
        language_question = get_response_message(completion_question)

        questions_and_answers = []
        questions_and_answers.append(get_question_with_context(self.current_question, self.context))

        # Insert at the end to make it more accurate
        questions_and_answers.append({"role": "system", "content": SYSTEM_MESSAGE + language_question})
        progress_message("Prompt is sent to AzureAI, waiting for response...")
        completion = get_chat_completion_gpt(self.client, questions_and_answers)
        progress_message("Answer received from AzureAI, saving results...")
        response_message = get_response_message(completion)
        return response_message
