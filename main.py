# main.py
#!/usr/bin/env python3
"""
RPA Invoice Processing System
Automazione completa elaborazione fatture con OCR e AI
"""

import sys
from pathlib import Path
import structlog
import schedule
import time
from datetime import datetime

from config import settings
from rpa.invoice_processor import InvoiceProcessorRPA
from integrations.email_monitor import EmailInvoiceMonitor
from integrations.azure_storage import AzureStorageManager
from database.invoice_repository import InvoiceRepository

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

class InvoiceRPAApplication:
    """Applicazione principale RPA"""
    
    def __init__(self):
        self.logger = logger.bind(app="InvoiceRPA")
        
        self.processor = InvoiceProcessorRPA()
        self.email_monitor = EmailInvoiceMonitor()
        self.storage = AzureStorageManager()
        self.repository = InvoiceRepository()
    
    def process_new_invoices(self):
        """Job principale: elabora nuove fatture"""
        self.logger.info("=== Inizio ciclo elaborazione ===")
        
        try:
            # 1. Fetch da email
            invoice_files = self.email_monitor.fetch_new_invoices()
            
            if not invoice_files:
                self.logger.info("Nessuna nuova fattura")
                return
            
            # 2. Elabora batch
            results = self.processor.process_batch(
                invoice_files,
                max_workers=settings.BATCH_SIZE // 3
            )
            
            # 3. Salva risultati
            for result in results:
                if result['status'] == 'success':
                    self._handle_successful_invoice(result)
                else:
                    self._handle_failed_invoice(result)
            
            # 4. Report
            stats = self.processor.get_stats()
            self.logger.info("Ciclo completato", **stats)
            
        except Exception as e:
            self.logger.error("Errore ciclo elaborazione", error=str(e), exc_info=True)
    
    def _handle_successful_invoice(self, result: Dict):
        """Gestisce fattura elaborata con successo"""
        invoice_data = result['invoice_data']
        file_path = Path(result['file_path'])
        
        try:
            # Upload su Azure Storage
            blob_url = self.storage.upload_invoice(
                file_path,
                invoice_data['invoice_number'],
                metadata={
                    'supplier': invoice_data['supplier_name'],
                    'amount': str(invoice_data['total_amount']),
                    'date': invoice_data['invoice_date']
                }
            )
            
            # Salva su database
            self.repository.save_invoice({
                **invoice_data,
                'blob_url': blob_url,
                'processing_status': 'completed' if not invoice_data['requires_manual_review'] else 'review',
                'processed_at': datetime.now().isoformat()
            })
            
            # Archive file locale
            archive_path = settings.ARCHIVE_DIR / file_path.name
            file_path.rename(archive_path)
            
            self.logger.info(
                "Fattura salvata",
                invoice_number=invoice_data['invoice_number'],
                requires_review=invoice_data['requires_manual_review']
            )
            
        except Exception as e:
            self.logger.error(
                "Errore salvataggio fattura",
                invoice=invoice_data.get('invoice_number'),
                error=str(e)
            )
    
    def _handle_failed_invoice(self, result: Dict):
        """Gestisce fattura con errori"""
        file_path = Path(result['file_path'])
        
        # Salva su DB come failed
        self.repository.save_invoice({
            'file_name': file_path.name,
            'processing_status': 'failed',
            'error_message': result['error'],
            'processed_at': datetime.now().isoformat()
        })
        
        # Sposta in cartella errori
        error_path = settings.ARCHIVE_DIR / "errors" / file_path.name
        error_path.parent.mkdir(exist_ok=True)
        file_path.rename(error_path)
    
    def run_once(self, input_path: Path = None):
        """Esegui singola elaborazione (per testing)"""
        if input_path:
            result = self.processor.process_invoice(input_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            self.process_new_invoices()
    
    def run_scheduler(self):
        """Avvia scheduler per elaborazione automatica"""
        self.logger.info("Avvio scheduler RPA")
        
        # Schedule ogni 15 minuti
        schedule.every(15).minutes.do(self.process_new_invoices)
        
        # Schedule report giornaliero
        schedule.every().day.at("18:00").do(self._send_daily_report)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def _send_daily_report(self):
        """Invia report giornaliero"""
        stats = self.processor.get_stats()
        
        self.logger.info("Report giornaliero", **stats)
        
        # TODO: Invia email con report

def main():
    """Entry point"""
    app = InvoiceRPAApplication()
    
    if len(sys.argv) > 1:
        # Modalità test: elabora singolo file
        input_file = Path(sys.argv[1])
        if not input_file.exists():
            print(f"File non trovato: {input_file}")
            sys.exit(1)
        
        app.run_once(input_file)
    else:
        # Modalità produzione: scheduler automatico
        app.run_scheduler()

if __name__ == "__main__":
    main()
