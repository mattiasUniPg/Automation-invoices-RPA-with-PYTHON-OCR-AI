# integrations/email_monitor.py
from exchangelib import (
    Credentials, Account, DELEGATE,
    FileAttachment, Message
)
from pathlib import Path
from typing import List
import structlog

from config import settings

logger = structlog.get_logger()

class EmailInvoiceMonitor:
    """Monitor casella email per nuove fatture"""
    
    def __init__(self):
        self.logger = logger.bind(component="EmailMonitor")
        
        credentials = Credentials(
            username=settings.EXCHANGE_EMAIL,
            password=settings.EXCHANGE_PASSWORD
        )
        
        self.account = Account(
            primary_smtp_address=settings.EXCHANGE_EMAIL,
            credentials=credentials,
            autodiscover=True,
            access_type=DELEGATE
        )
    
    def fetch_new_invoices(self) -> List[Path]:
        """Scarica nuove fatture dalla casella email"""
        
        self.logger.info("Controllo nuove fatture in arrivo")
        
        inbox = self.account.inbox / settings.INBOX_FOLDER
        processed_folder = self.account.inbox / settings.PROCESSED_FOLDER
        
        invoice_files = []
        
        # Fetch unread emails
        for item in inbox.filter(is_read=False):
            if not isinstance(item, Message):
                continue
            
            self.logger.info(
                "Email ricevuta",
                subject=item.subject,
                from_email=item.sender.email_address
            )
            
            # Process attachments
            for attachment in item.attachments:
                if isinstance(attachment, FileAttachment):
                    if self._is_invoice_file(attachment.name):
                        file_path = self._save_attachment(attachment)
                        invoice_files.append(file_path)
            
            # Mark as read and move
            item.is_read = True
            item.save()
            item.move(processed_folder)
        
        self.logger.info("Fatture scaricate", count=len(invoice_files))
        
        return invoice_files
    
    def _is_invoice_file(self, filename: str) -> bool:
        """Verifica se file è una fattura"""
        invoice_keywords = ['fattura', 'invoice', 'ft_', 'inv_']
        valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
        
        filename_lower = filename.lower()
        
        has_keyword = any(kw in filename_lower for kw in invoice_keywords)
        has_valid_ext = any(filename_lower.endswith(ext) for ext in valid_extensions)
        
        return has_keyword and has_valid_ext
    
    def _save_attachment(self, attachment: FileAttachment) -> Path:
        """Salva allegato su disco"""
        file_path = settings.TEMP_DIR / attachment.name
        
        with open(file_path, 'wb') as f:
            f.write(attachment.content)
        
        self.logger.debug("Attachment salvato", file=str(file_path))
        
        return file_path

# integrations/azure_storage.py
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceExistsError

class AzureStorageManager:
    """Gestione storage Azure per archiviazione fatture"""
    
    def __init__(self):
        self.logger = logger.bind(component="AzureStorage")
        
        self.blob_service = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        
        self.container_client = self.blob_service.get_container_client(
            settings.BLOB_CONTAINER_NAME
        )
        
        # Crea container se non esiste
        try:
            self.container_client.create_container()
        except ResourceExistsError:
            pass
    
    def upload_invoice(
        self, 
        file_path: Path,
        invoice_number: str,
        metadata: Dict = None
    ) -> str:
        """Upload fattura su Azure Blob Storage"""
        
        # Organizza per anno/mese
        now = datetime.now()
        blob_name = f"{now.year}/{now.month:02d}/{invoice_number}_{file_path.name}"
        
        blob_client = self.container_client.get_blob_client(blob_name)
        
        with open(file_path, 'rb') as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata=metadata or {}
            )
        
        self.logger.info("Fattura archiviata", blob=blob_name)
        
        return blob_client.url
      //*  Integrazione Email e Storage */
