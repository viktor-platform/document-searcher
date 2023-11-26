import markdown


def list_to_html_string(input_list: list[str]):
    """Convert sources to nice-looking html"""
    html_string = "<html>\n"
    html_string += "<body>\n"
    html_string += "<strong>The following documents are currently included in the search:</strong>\n"
    html_string += "<ul>\n"
    for item in input_list:
        html_string += f"<li>{item}</li>\n"
    html_string += "</ul>\n"
    html_string += "</body>\n"
    html_string += "</html>\n"
    return html_string


def generate_html_code(question: str, answer: str, metadata: list[dict], context_history: list[str]):
    """Present the results in a nice-looking html"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Chat Conversation</title>
        <style>
            body{
                font-family: "Roboto","Helvetica","Arial", sans-serif;
                font-size: 0.875rem;
                line-height: 1.43;
                letter-spacing: 0.01071em;
                color: #4e4b66;
            }

            .user {
                display: block;
                font-weight: bold;
                font-style: italic;
                background: #F7F7FC;
                border-radius: 4px;
            }

            .user p {
                padding: 1em;
            }
            
            .message {
                margin: 10px;
                border: 1px solid #ccc;
                padding: 10px;
            }

            .content {
              display: none;
              margin-top: 1px;
              padding: 1px;
            }
            
            .summary {
                cursor: pointer;
            }

            details[open] .content {
              display: block;
            }

            pre {
                background-color: #f8f9fa;
                border-radius: 4px;
                overflow-x: auto;
            }
            blockquote {
                background-color: #f2f2f2; /* Set the background color to grey */
                padding: 10px; /* Add some padding to give it some space */
            }        
        </style>
    </head>
    <body>
        <div class="conversation">
    """
    statement = f"## {question}\n" + answer
    content = markdown.markdown(statement)

    html += f'<div class="message">{content}</div>\n'

    html += "<blockquote>"
    statement = f"## Sources"
    content = markdown.markdown(statement)
    html += f'<div class="message">{content}</div>\n'
    for context, metadata_source in zip(context_history, metadata):
        statement = f"Page {metadata_source['page_number']} - Document {metadata_source['source']}"
        content = markdown.markdown(context)
        html += '<details class="message"> <summary>{}</summary> <div class="content">{}</div> </details>\n'.format(
            statement, content
        )
    html += "</blockquote>"

    html += """
        </div>
        <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/base16/chalk.min.css">
        <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
    </body>
    </html>
    """
    return html
