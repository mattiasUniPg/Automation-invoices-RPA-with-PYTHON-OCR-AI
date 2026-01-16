# database/invoice_repository.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import structlog

from config import settings

logger = structlog.get_logger()
Base = declarative_base()

class Invoice(Base):
    """Modello database fattura"""
    __tablename__ = 'invoices'
    
    id = Column(Integer, primary_key=True)
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    invoice_date = Column(String(10))
    
    supplier_name = Column(String(200))
    supplier_vat = Column(String(11), index=True)
    
    customer_name = Column(String(200))
    customer_vat = Column(String(11), index=True)
    
    subtotal = Column(Float)
    vat_amount = Column(Float)
    total_amount = Column(Float)
    currency = Column(String(3), default='EUR')
    
    ocr_confidence = Column(Float)
    ai_validation_score = Column(Float)
    
    processing_status = Column(String(20), index=True)  # completed, review, failed
    requires_manual_review = Column(Boolean, default=False)
    
    blob_url = Column(String(500))
    file_name = Column(String(255))
    
    validation_notes = Column(Text)
    error_message = Column(Text)
    
    processed_at = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)

class InvoiceRepository:
    """Repository per gestione fatture nel database"""
    
    def __init__(self):
        self.logger = logger.bind(component="DBRepository")
        
        self.engine = create_engine(settings.DB_CONNECTION_STRING)
        Base.metadata.create_all(self.engine)
        
        self.Session = sessionmaker(bind=self.engine)
    
    def save_invoice(self, invoice_data: Dict) -> int:
        """Salva fattura nel database"""
        session = self.Session()
        
        try:
            invoice = Invoice(
                invoice_number=invoice_data.get('invoice_number', 'N/A'),
                invoice_date=invoice_data.get('invoice_date'),
                supplier_name=invoice_data.get('supplier_name'),
                supplier_vat=invoice_data.get('supplier_vat'),
                customer_name=invoice_data.get('customer_name'),
                customer_vat=invoice_data.get('customer_vat'),
                subtotal=invoice_data.get('subtotal'),
                vat_amount=invoice_data.get('vat_amount'),
                total_amount=invoice_data.get('total_amount'),
                currency=invoice_data.get('currency', 'EUR'),
                ocr_confidence=invoice_data.get('ocr_confidence'),
                ai_validation_score=invoice_data.get('ai_validation_score'),
                processing_status=invoice_data.get('processing_status', 'completed'),
                requires_manual_review=invoice_data.get('requires_manual_review', False),
                blob_url=invoice_data.get('blob_url'),
                file_name=invoice_data.get('file_name'),
                validation_notes=json.dumps(invoice_data.get('validation_notes', [])),
                error_message=invoice_data.get('error_message')
            )
            
            session.add(invoice)
            session.commit()
            
            self.logger.info("Fattura salvata nel database", id=invoice.id)
            
            return invoice.id
            
        except Exception as e:
            session.rollback()
            self.logger.error("Errore salvataggio database", error=str(e))
            raise
        finally:
            session.close()
