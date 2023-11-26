from viktor.parametrization import BooleanField
from viktor.parametrization import ChildEntityManager
from viktor.parametrization import SetParamsButton
from viktor.parametrization import Tab
from viktor.parametrization import Text
from viktor.parametrization import TextAreaField
from viktor.parametrization import ViktorParametrization


class Parametrization(ViktorParametrization):
    """Parametrization class for document searcher"""

    input = Tab("Input")
    input.welcome_text = Text(
        "# \U0001F50D Document searcher  \n"
        "Welcome to the VIKTOR document searcher. "
        "With this app, you can easily find information in your PDF documents. Your question will be answered using "
        "the power of AI."
    )
    input.pdf_uploaded = Text("**Step 1:** Upload documents [PDF], by clicking '+ Create PDF'", flex=100)
    input.pdf_manager = ChildEntityManager("ProcessPDF")
    input.text_step_2 = Text("**Step 2**: Submit your documents by clicking on the button below")
    input.set_embeddings_button = SetParamsButton("\u2705 Submit documents", "set_embeddings", flex=45, longpoll=True)
    input.question = TextAreaField(
        "**Step 3:** Ask your question here",
        flex=100,
        description="Any language is allowed, the app will answer in the same language as your question.",
    )
    input.step_4_text = Text("**Step 4:** Get your result by clicking 'Update' in the lower-right corner of this app.")

    input.embeddings_are_set = BooleanField("embeddings_are_set", default=False, visible=False)

    privacy_info = Tab("\U0001f4da Privacy & Usage info")
    privacy_info.disclaimer_text = Text(
        """
## Why we offer this app
As a user of the VIKTOR platform we want you to see the power of it. The document searcher can be used by many people 
within your organisation and is a fun way to get introduced to VIKTOR. 
        
## Usage

This app uses AI to help you ask questions about PDFs you upload. But remember, 
the AI can only read text. It can't understand pictures, a scanned document will not work. This can be easily checked 
by opening the PDF and trying to select text. If you cannot select the text, this document will not work in this 
application. \n
After you successfully upload a PDF, you can ask questions about it. The app will answer based on what's in 
the PDF. It'll also tell you the source of the answer, like the document name and page number. This is especially 
convenient when you're handling many documents, like in the planning phase of a project. You can also save your work in 
the app. Once you upload a PDF and hit the save button (top-right corner), your team can see it if they use the app too. 

## Privacy

The admins of your VIKTOR environment are in full control of who can use the app. No one else can see the documents 
uploaded in the app unless they have permission. And even then, only if you save the documents in the app.  

Please note that no VIKTOR employees have access to your data. Access to the uploaded documents can only be achieved by 
being added to the application in your environment by your admins.  

The data is not used to train the model or for other purposes, nor by OpenAI, AzureAI, Microsoft or VIKTOR.
"""
    )
