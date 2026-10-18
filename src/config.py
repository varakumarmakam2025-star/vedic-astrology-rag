import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PRIMARY_MODEL = "gemini-3.8-flash"  # Updated to the currently available model version
TOP_K_RETRIEVAL = 3
CHROMA_COLLECTION_NAME = "enterprise_docs"
DATA_PATH = "enterprise_docs/stripe_api_docs.txt"

class Settings:
    def __init__(self):
        self.GEMINI_API_KEY = GEMINI_API_KEY
        self.PRIMARY_MODEL = PRIMARY_MODEL
        self.TOP_K_RETRIEVAL = TOP_K_RETRIEVAL
        self.CHROMA_COLLECTION_NAME = CHROMA_COLLECTION_NAME
        self.DATA_PATH = DATA_PATH

settings = Settings()

if not settings.GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in your .env file!")