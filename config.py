# config.py
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Configurazione centralizzata"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    ARCHIVE_DIR: Path = BASE_DIR / "archive"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # OCR Configuration
    TESSERACT_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSERACT_LANG: str = "ita+eng"
    OCR_DPI: int = 300
    OCR_CONFIDENCE_THRESHOLD: float = 70.0
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4"
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    
    # Email Configuration
    EXCHANGE_EMAIL: str
    EXCHANGE_PASSWORD: str
    EXCHANGE_SERVER: str
    INBOX_FOLDER: str = "Fatture"
    PROCESSED_FOLDER: str = "Fatture/Elaborate"
    
    # Database
    DB_CONNECTION_STRING: str
    
    # Azure Storage
    AZURE_STORAGE_CONNECTION_STRING: str
    BLOB_CONTAINER_NAME: str = "invoices"
    
    # Business Rules
    VAT_RATE: float = 0.22
    MAX_INVOICE_AMOUNT: float = 100000.0
    AUTO_APPROVE_THRESHOLD: float = 5000.0
    
    # Processing Options
    BATCH_SIZE: int = 10
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# Crea directories necessarie
for directory in [settings.TEMP_DIR, settings.ARCHIVE_DIR, settings.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
