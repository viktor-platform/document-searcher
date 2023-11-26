import pickle

import numpy as np
import pandas as pd
from viktor import UserError
from viktor import UserMessage
from viktor import ViktorController
from viktor.api_v1 import API
from viktor.core import File
from viktor.core import Storage
from viktor.result import SetParamsResult
from viktor.views import WebResult
from viktor.views import WebView

from app.project.parametrization import Parametrization

from ..AI_search.chat_view import generate_html_code
from ..AI_search.chat_view import list_to_html_string
from ..AI_search.retrieval_assistant import RetrievalAssistant


class Controller(ViktorController):
    """Controller class for Document searcher app"""

    label = "Documents"
    parametrization = Parametrization
    children = ["ProcessPDF"]
    show_children_as = "Table"

    def set_embeddings(self, params, entity_id, **kwargs):
        """Takes in one or multiple PDF documents and returns a single chunked, embedded file. The metadata for page
        number and document name is included in the embedded file. The embedded file is saved to storage.
        """
        df_list = []
        pdf_names = []
        current_entity = API().get_entity(entity_id)
        pdf_entities = current_entity.children()
        if not pdf_entities:
            raise UserError("Please upload your PDF documents first")
        for pdf_file_entity in pdf_entities:
            UserMessage.info(f"Receiving data for {pdf_file_entity.name}")
            df = pickle.loads(Storage().get("pdf_storage", scope="entity", entity=pdf_file_entity).getvalue_binary())
            df_list.append(df)
            pdf_names.append(pdf_file_entity.name)
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df["embeddings"] = combined_df["embeddings"].apply(np.array)
        UserMessage.success("Document succesfully embedded!")
        pdf_names_str = list_to_html_string(pdf_names)
        viktor_file = File()
        with viktor_file.open_binary() as f:
            combined_df.to_pickle(f)
        Storage().set("embeddings_storage", viktor_file, scope="entity")
        Storage().set("list_of_files", File.from_data(pdf_names_str), scope="entity")
        return SetParamsResult({"input": {"embeddings_are_set": True}})

    @WebView("Conversation", duration_guess=5)
    def conversation(self, params, **kwargs):
        """View for showing the questions, answers and sources to the user."""
        if not params.input.embeddings_are_set:
            raise UserError("Please embed the uploaded PDF file first, by clicking 'Submit document(s)'.")
        df = pickle.loads(Storage().get("embeddings_storage", scope="entity").getvalue_binary())
        retrieval_assistant = RetrievalAssistant(params.input.question, df)
        answer = retrieval_assistant.ask_assistant()
        html = generate_html_code(
            params.input.question, answer, retrieval_assistant.metadata_list, retrieval_assistant.context_list
        )
        return WebResult(html=html)

    @WebView("Document list", duration_guess=1)
    def document_list_view(self, params, **kwargs):
        """View for showing which documents are currently embedded in storage and used to answer the questions."""
        if not params.input.embeddings_are_set:
            pdf_names_str = "No documents have been uploaded or submitted yet."
        else:
            pdf_names_str = Storage().get("list_of_files", scope="entity").getvalue()
        return WebResult(html=pdf_names_str)
