import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.lib.azure import AzureOpenAI
from pypdf import PdfReader
from viktor import File
from viktor import ParamsFromFile
from viktor import ViktorController
from viktor.core import Storage
from viktor.core import UserMessage

from app.AI_search.config import MAX_RETRIES
from app.AI_search.helper_functions import get_API_key
from app.AI_search.helper_functions import get_embedding


class Controller(ViktorController):
    """Controller class for processing the PDF files and embedding the text within the PDF files."""

    label = "PDF"

    @ParamsFromFile(file_types=[".pdf"])
    def process_file(self, pdf_file: File, entity_name, **kwargs) -> dict:
        """Process the PDF file when it is first uploaded"""

        # Splitter is used to chunk the document, so the chunk size doesn't become too big for AzureAI
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n"])
        documents = []
        with pdf_file.open_binary() as pdf_opened:
            reader = PdfReader(pdf_opened)
            for page_number, page in enumerate(reader.pages):
                page_extracted_text = page.extract_text()
                page_extracted_text_split = splitter.split_text(page_extracted_text)
                for split_text in page_extracted_text_split:
                    documents.append(
                        Document(
                            page_content=split_text,
                            metadata={
                                "page_number": page_number + 1,
                                "source": entity_name.split(".")[0],
                            },
                        )
                    )

        # Embed chunks
        embedded_documents = []
        API_KEY, ENDPOINT = get_API_key()
        client = AzureOpenAI(
            api_key=API_KEY, api_version="2023-10-01-preview", azure_endpoint=ENDPOINT, max_retries=MAX_RETRIES
        )

        for document in documents:
            UserMessage.info(f"Embedding page {document.metadata['page_number']} for " f"{document.metadata['source']}")
            embedding = get_embedding(client, document.page_content)
            embedded_documents.append(
                {
                    "text": document.page_content,
                    "embeddings": embedding,
                    "page_number": document.metadata["page_number"],
                    "source": document.metadata["source"],
                }
            )
        df = pd.DataFrame(embedded_documents)
        vikor_file = File()
        with vikor_file.open_binary() as f:
            df.to_pickle(f)
        Storage().set("pdf_storage", vikor_file, scope="entity")
        return {}
