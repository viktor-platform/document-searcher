COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0
RETRIES = 3
N_CONTEXT = 5

RETRY_MESSAGE = [
    "My apologies. I was not concentrating. Let me retry your request...",
    "Thank you for your patience. Today I'm not performing as well as I should. Let's try again...",
    "It seems that I am struggling more than I should. Let's see if I can better understand your case..."
    "Uhm... That's weird... I thought I gave you an answer... Let's try again...",
]

SYSTEM_MESSAGE = """You are a helpful assistant and answer the questions, based on the provided context."""
