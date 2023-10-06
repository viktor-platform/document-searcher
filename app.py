import pandas as pd
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from viktor import UserError
from viktor import UserMessage
from viktor import ViktorController
from viktor import progress_message
from viktor.core import File
from viktor.core import Storage
from viktor.parametrization import BooleanField
from viktor.parametrization import IsTrue
from viktor.parametrization import Lookup
from viktor.parametrization import MultiFileField
from viktor.parametrization import NumberField
from viktor.parametrization import OptionField
from viktor.parametrization import OutputField
from viktor.parametrization import SetParamsButton
from viktor.parametrization import Tab
from viktor.parametrization import Table
from viktor.parametrization import Text
from viktor.parametrization import TextAreaField
from viktor.parametrization import ViktorParametrization
from viktor.result import SetParamsResult
from viktor.views import WebResult
from viktor.views import WebView

from config import EMBEDDINGS_MODEL
from helper_functions import VIKTOR_file_to_df
from helper_functions import df_to_VIKTOR_csv_file
from helper_functions import generate_html_code
from helper_functions import list_to_html_string
from memory_error_functions import NO_MEMORY_ERROR_MESSAGE
from memory_error_functions import _get_document_names
from memory_error_functions import check_if_page_is_excluded
from memory_error_functions import get_memory_error
from retrieval_assistant import RetrievalAssistant
from retrieval_assistant import get_API_key


class Parametrization(ViktorParametrization):
    """Parametrization class for document searcher"""
    input = Tab("Input")
    input.welcome_text = Text(
        "# \U0001F50D Document searcher  \n"
        "Welcome to the VIKTOR document searcher. "
        "With this app, you can easily find information in your PDF documents. Your question will be answered using "
        "the power of AzureAI."
    )
    input.pdf_uploaded = MultiFileField("**Step 1:** Upload documents [PDF]", flex=100, file_types=[".pdf"])
    input.text_step_2 = Text("**Step 2**: Submit your documents by clicking on the button below")
    input.set_embeddings_button = SetParamsButton("\u2705 Submit documents", "set_embeddings", flex=45, longpoll=True)
    input.question = TextAreaField(
        "**Step 3:** Ask your question here",
        flex=100,
    )
    input.step_4_text = Text("**Step 4:** Get your result by clicking 'Update' in the lower-right corner of this app.")

    input.embeddings_are_set = BooleanField("embeddings_are_set", default=False, visible=False)
    input.out_of_memory_toggle = BooleanField("**Troubleshoot:** Out of memory? Click here to continue", flex=100)
    input.out_of_memory_text = Text(
        "A known issue within the app is that some PDF pages can cause the app to run out of "
        "memory. This happens for example when technical drawings are included in the PDF. If "
        "a memory error occured, below is shown at what page and in which document. Please exclude this "
        "page by entering the document name and page number in the table below.  \n"
        "If this doesn't work, try uploading less PDF files. An issue is known that the app doesn't work when too many "
        "files are uploaded. The app has been tested to work for around 10-15 pdf files, containing around "
        "800-1000 pages.",
        visible=IsTrue(Lookup("input.out_of_memory_toggle")),
    )
    input.memory_error_output = OutputField(
        "Memory error at:", value=get_memory_error, visible=IsTrue(Lookup("input.out_of_memory_toggle")), flex=100
    )
    input.exclude_pages_table = Table("Exclude pages from document",
                                      visible=IsTrue(Lookup("input.out_of_memory_toggle")))
    input.exclude_pages_table.document_name = OptionField("Document name", options=_get_document_names)
    input.exclude_pages_table.page_number = NumberField("Page number")

    privacy_info = Tab("\U0001f4da Privacy & Usage info")
    privacy_info.disclaimer_text = Text(
        """## Usage

This application uses a Large Language Model to make it possible to ask questions about an uploaded PDF. 
Note that the Large Language Model (LLM) can only process text-based information can not read images. That means that 
the PDF needs to be fully text-based, a picture of a document will not work.  

Once a suitable document is uploaded, all the text from the PDF is embedded (a sort of index). Then, questions can be 
asked regarding the document(s) and are answered based on those document(s). Additionally, also the source of the answer 
is provided, by name of the document and page number. This tool is especially useful when there are lots of documents, 
for example in the tender phase of a project.

This app includes save-functionality. Once a document is uploaded by a user and the session is saved with the button in
the top-right corner of the screen, this project (and thus the documents therein) is visible to all users who have 
access to this application. 

## Privacy

The admins of your VIKTOR environment are in ful control of who gets access to the app; just like any other app. No one 
can see the documents uploaded in the app. Only people with access to your app and only if one saves the documents in 
the app.  

The LLM model is hosted on by VIKTOR on AzureAI and we use our own tokens. This means VIKTOR pays all the costs 
for this enterprise service.  

Even though we host the model, no VIKTOR employees have access to your data through that hosting service. Access to the
documents uploaded can only be achieved by being added to the application in your environment by your admins.  

The data is also not used to train the model or for other purposes, nor by OpenAI, Microsoft or VIKTOR.
 
        """
    )


class Controller(ViktorController):
    """Controller class for Document searcher app"""

    label = "Documents"
    parametrization = Parametrization

    def set_embeddings(self, params, **kwargs):
        """Takes in one or multiple PDF documents and returns a single chunked, embedded file. The metadata for page
        number and document name is included in the embedded file. The embedded file is saved to storage.
        """
        if not params.input.pdf_uploaded:
            raise UserError("Please upload a PDF file first.")

        # Splitter is used to chunk the document, so the chunk size doesn't become too big for ChatGPT
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n"])

        # Read and chunk PDF
        documents = []
        pdf_information = {}
        for pdf_file in params.input.pdf_uploaded:
            with pdf_file.file.open_binary() as pdf_opened:
                with pdfplumber.open(pdf_opened) as pdf:
                    UserMessage.info(f"Pre-processing {pdf_file.filename}")
                    pdf_information[pdf_file.filename] = {"n_pages": len(pdf.pages), "pdf_file": pdf_file}

        for pdf_filename in pdf_information:
            pages_range = pdf_information[pdf_filename]["n_pages"]
            for page_number in range(pages_range):
                pdf_file = pdf_information[pdf_filename]["pdf_file"]
                with pdf_file.file.open_binary() as pdf_opened:
                    with pdfplumber.open(pdf_opened) as pdf:
                        if check_if_page_is_excluded(
                            page_number, pdf_file.filename, params.input.exclude_pages_table,
                                params.input.out_of_memory_toggle
                        ):
                            continue
                        page = pdf.pages[page_number]
                        progress_message(f"Setting up vector dataframe for...")
                        UserMessage.info(f"Reading page {page_number + 1}/{pages_range} from {pdf_file.filename}")
                        current_file_and_page = File.from_data(
                            f"Page number: {page_number + 1} - Filename: {pdf_file.filename}"
                        )
                        Storage().set("current_document_and_page", current_file_and_page, scope="entity")
                        page_text_split = splitter.split_text(page.extract_text())
                        for split_text in page_text_split:
                            documents.append(
                                Document(
                                    page_content=split_text,
                                    metadata={
                                        "source": pdf_file.filename,
                                        "page_number": page_number + 1,
                                        "n_pages": pdf_information[pdf_filename]["n_pages"],
                                    },
                                )
                            )
                        Storage().set("current_document_and_page", File.from_data(NO_MEMORY_ERROR_MESSAGE), scope="entity")
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
            UserMessage.info(
                f"Embedding page {document.metadata['page_number']}/{document.metadata['n_pages']} for "
                f"{document.metadata['source']}"
            )
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
        pdf_names_str = list_to_html_string(pdf_information.keys())
        Storage().set("embeddings_storage", csv_file, scope="entity")
        Storage().set("list_of_files", File.from_data(pdf_names_str), scope="entity")
        return SetParamsResult({"input": {"embeddings_are_set": True}})

    @WebView("Conversation", duration_guess=5)
    def conversation(self, params, **kwargs):
        """View for showing the questions, answers and sources to the user."""
        if not params.input.pdf_uploaded:
            raise UserError("Please upload a PDF file first.")
        elif not params.input.embeddings_are_set:
            raise UserError("Please embed the uploaded PDF file first, by clicking 'Submit document(s)'.")
        else:
            embeddings_file = Storage().get("embeddings_storage", scope="entity")
            df = VIKTOR_file_to_df(embeddings_file)
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
