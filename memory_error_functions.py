from munch import Munch
from viktor.core import Storage
from viktor.parametrization import Table

NO_MEMORY_ERROR_MESSAGE = "No out of memory detected"


def get_memory_error(**kwargs):
    """Check if memory occured, and if so return the document name and page number on which the error occured"""
    try:
        storage = Storage().get("current_document_and_page", scope="entity")
        document_name_and_page = storage.getvalue()
    except FileNotFoundError:
        document_name_and_page = NO_MEMORY_ERROR_MESSAGE
    return document_name_and_page


def _get_document_names(params: Munch, **kwargs):
    """Returns the names of all uploaded documents"""
    pdf_file_names = []
    for pdf_file in params.input.pdf_uploaded:
        pdf_file_names.append(pdf_file.filename)
    return pdf_file_names


def check_if_page_is_excluded(
    page_number: int, filename: str, exclude_pages_table: Table, out_of_memory_toggle: bool
) -> bool:
    """Checks if page in document is excluded from reading the PDF."""
    if out_of_memory_toggle:
        for row in exclude_pages_table:
            page_number_index = page_number + 1
            if page_number_index == row.page_number and filename == row.document_name:
                return True
