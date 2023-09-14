import io

import pandas as pd
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from viktor import UserError
from viktor import UserMessage
from viktor import ViktorController
from viktor import progress_message
from viktor.core import Storage, File
from viktor.parametrization import MultiFileField, BooleanField
from viktor.parametrization import SetParamsButton
from viktor.parametrization import Text
from viktor.parametrization import TextAreaField
from viktor.parametrization import ViktorParametrization
from viktor.result import SetParamsResult
from viktor.views import WebResult
from viktor.views import WebView

from config import EMBEDDINGS_MODEL
from helper_functions import generate_html_code, df_to_VIKTOR_csv_file, VIKTOR_file_to_df, list_to_html_string
from retrieval_assistant import RetrievalAssistant, get_API_key


class Parametrization(ViktorParametrization):
    welcome_text = Text(
        "# \U0001F50D Document searcher  \n"
        "Welcome to the VIKTOR document searcher. "
        "With this app, you can easily find information in your PDF documents. Your question will be answered using "
        "the power of AzureAI."
    )
    pdf_uploaded = MultiFileField("**Step 1:** Upload documents [PDF]", flex=100, file_types=[".pdf"])
    set_embeddings_button = SetParamsButton("**Step 2:** Submit documents \u2705", "set_embeddings", flex=45, longpoll=True)
    question = TextAreaField(
        "**Step 3:** Ask your question here",
        flex=100,
    )
    step_4_text = Text("**Step 4:** Get your result by clicking 'Update' in the lower-right corner of this app.")
    disclaimer_text = Text("**Privacy:** All your data will remain strictly confidential, and it will "
                           "not be used to train any model.")
    embeddings_are_set = BooleanField("embeddings_are_set", default=False, visible=False)


class Controller(ViktorController):
    label = "Documents"
    parametrization = Parametrization

    def set_embeddings(self, params, **kwargs):
        """Takes in one or multiple PDF documents and returns a single chunked, embedded file. The metadata for page
        number and document name is included in the embedded file. The embedded file is saved to storage.
        """
        if not params.pdf_uploaded:
            raise UserError("Please upload a PDF file first.")

        # Splitter is used to chunk the document, so the chunk size doesn't become too big for ChatGPT
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n"])

        # Read and chunk PDF
        documents = []
        pdf_names = []
        for pdf_file in params.pdf_uploaded:
            with pdfplumber.open(io.BytesIO(pdf_file.file.getvalue_binary())) as pdf:
                n_pages = len(pdf.pages)
                pdf_names.append(pdf_file.filename)
                for page_number, page in enumerate(pdf.pages, start=1):
                    progress_message(f"Setting up vector dataframe for...")
                    UserMessage.info(f"Reading page {page_number}/{n_pages} from {pdf_file.filename}")
                    page_text_split = splitter.split_text(page.extract_text(stream=True))
                    for split_text in page_text_split:
                        documents.append(
                            Document(
                                page_content=split_text,
                                metadata={"source": pdf_file.filename, "page_number": page_number},
                            )
                        )
                    page.flush_cache()

        # Embed chunks
        API_KEY, ENDPOINT = get_API_key()
        embeddings = OpenAIEmbeddings(
                openai_api_key=API_KEY,
                deployment=EMBEDDINGS_MODEL,
                model=EMBEDDINGS_MODEL,
                openai_api_base=ENDPOINT,
                openai_api_type="azure",
            )
        embedded_documents = []
        for document in documents:
            UserMessage.info(f"Embedding page {document.metadata['page_number']}/{n_pages} for "
                             f"{document.metadata['source']}")
            embedded_documents.append(
                {
                    "text": document.page_content,
                    "embeddings": embeddings.embed_query(document.page_content),
                    "page_number": document.metadata["page_number"],
                    "source": document.metadata["source"],
                }
            )

        # Save embeddings to storage
        UserMessage.success("Finishing up...")
        df = pd.DataFrame(embedded_documents)
        csv_file = df_to_VIKTOR_csv_file(df)
        UserMessage.success("Document succesfully embedded!")
        pdf_names_str = list_to_html_string(pdf_names)
        Storage().set("embeddings_storage", csv_file, scope='entity')
        Storage().set("list_of_files", File.from_data(pdf_names_str), scope='entity')
        return SetParamsResult({"embeddings_are_set": True})

    @WebView("Conversation", duration_guess=5)
    def conversation(self, params, **kwargs):
        """View for showing the questions, answers and sources to the user."""
        if not params.pdf_uploaded:
            raise UserError("Please upload a PDF file first.")
        elif not params.embeddings_are_set:
            raise UserError("Please embed the uploaded PDF file first, by clicking 'Submit document(s)'.")
        else:
            embeddings_file = Storage().get("embeddings_storage", scope="entity")
            df = VIKTOR_file_to_df(embeddings_file)
            retrieval_assistant = RetrievalAssistant(params.question, df)
            answer = retrieval_assistant.ask_assistant()
            html = generate_html_code(
                params.question, answer, retrieval_assistant.metadata_list, retrieval_assistant.context_list
            )
        return WebResult(html=html)

    @WebView("Document list", duration_guess=1)
    def document_list_view(self, params, **kwargs):
        """View for showing which documents are currently embedded in storage and used to answer the questions."""
        if not params.embeddings_are_set:
            pdf_names_str = "No documents have been uploaded or submitted yet."
        else:
            pdf_names_str = Storage().get("list_of_files", scope='entity').getvalue()
        return WebResult(html=pdf_names_str)
