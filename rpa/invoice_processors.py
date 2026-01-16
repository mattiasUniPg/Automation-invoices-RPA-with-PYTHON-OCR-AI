# rpa/invoice_processor.py
from pathlib import Path
from typing import Optional, Dict, List
import shutil
from datetime import datetime
import structlog
from pdf2image import convert_from_path

from preprocessing.image_processor import ImagePreprocessor
from ocr.ocr_engine import InvoiceOCREngine
from ai.azure_openai_validator import AzureOpenAIValidator, InvoiceData
from config import settings

logger = structlog.get_logger()

class InvoiceProcessorRPA:
    """Orchestratore principale pipeline RPA fatture"""
    
    def __init__(self):
        self.logger = logger.bind(component="RPAProcessor")
        
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = InvoiceOCREngine()
        self.ai_validator = AzureOpenAIValidator()
        
        # Stats
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'manual_review': 0
        }
    
    def process_invoice(
        self, 
        file_path: Path,
        save_debug_images: bool = True
    ) -> Dict:
        """
        Pipeline completa elaborazione fattura
        
        Args:
            file_path: Path al file PDF o immagine
            save_debug_images: Se salvare immagini intermedie
            
        Returns:
            Dizionario con risultati elaborazione
        """
        self.logger.info("Inizio elaborazione fattura", file=str(file_path))
        start_time = datetime.now()
        
        try:
            # Step 1: Conversione PDF → Immagine (se necessario)
            image_path = self._prepare_image(file_path)
            
            # Step 2: Preprocessing immagine
            if save_debug_images:
                preprocessed_path = settings.TEMP_DIR / f"{file_path.stem}_preprocessed.png"
            else:
                preprocessed_path = None
            
            preprocessed_img = self.preprocessor.preprocess_invoice(
                image_path,
                preprocessed_path
            )
            
            # Step 3: Estrazione testo OCR
            ocr_result = self.ocr_engine.extract_text(preprocessed_img)
            
            # Step 4: Estrazione campi strutturati
            extracted_fields = self.ocr_engine.extract_invoice_fields(
                ocr_result.text,
                ocr_result.word_data
            )
            
            # Convert InvoiceField to dict for AI
            fields_dict = {
                k: v.value for k, v in extracted_fields.items()
            }
            
            # Step 5: Validazione e strutturazione con AI
            validated_invoice = self.ai_validator.validate_and_structure_invoice(
                ocr_text=ocr_result.text,
                extracted_fields=fields_dict,
                ocr_confidence=ocr_result.confidence
            )
            
            # Step 6: Semantic similarity check
            similarity = self.ai_validator.semantic_similarity_check(
                ocr_result.text,
                validated_invoice
            )
            
            if similarity < 0.6:
                validated_invoice.requires_manual_review = True
                validated_invoice.validation_notes.append(
                    f"Bassa coerenza semantica: {similarity:.2f}"
                )
            
            # Step 7: Salva risultati
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'invoice_data': validated_invoice.model_dump(),
                'ocr_confidence': ocr_result.confidence,
                'semantic_similarity': similarity,
                'processing_time_seconds': processing_time,
                'requires_manual_review': validated_invoice.requires_manual_review,
                'file_path': str(file_path)
            }
            
            self.stats['processed'] += 1
            if validated_invoice.requires_manual_review:
                self.stats['manual_review'] += 1
            else:
                self.stats['successful'] += 1
            
            self.logger.info(
                "Fattura elaborata con successo",
                invoice_number=validated_invoice.invoice_number,
                amount=validated_invoice.total_amount,
                processing_time=f"{processing_time:.2f}s",
                requires_review=validated_invoice.requires_manual_review
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Errore elaborazione fattura",
                file=str(file_path),
                error=str(e),
                exc_info=True
            )
            
            self.stats['processed'] += 1
            self.stats['failed'] += 1
            
            return {
                'status': 'error',
                'error': str(e),
                'file_path': str(file_path)
            }
    
    def _prepare_image(self, file_path: Path) -> Path:
        """Converte PDF in immagine se necessario"""
        
        if file_path.suffix.lower() == '.pdf':
            self.logger.debug("Conversione PDF in immagine")
            
            # Converti prima pagina PDF
            images = convert_from_path(
                str(file_path),
                dpi=settings.OCR_DPI,
                first_page=1,
                last_page=1
            )
            
            image_path = settings.TEMP_DIR / f"{file_path.stem}.png"
            images[0].save(image_path, 'PNG')
            
            return image_path
        
        return file_path
    
    def process_batch(
        self, 
        files: List[Path],
        max_workers: int = 3
    ) -> List[Dict]:
        """Elaborazione batch con parallelizzazione"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        self.logger.info("Inizio elaborazione batch", count=len(files))
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_invoice, f): f 
                for f in files
            }
            
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(
                        "Errore elaborazione batch",
                        file=str(file),
                        error=str(e)
                    )
                    results.append({
                        'status': 'error',
                        'error': str(e),
                        'file_path': str(file)
                    })
        
        self.logger.info(
            "Batch completato",
            **self.stats
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche elaborazione"""
        if self.stats['processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['processed']) * 100
            review_rate = (self.stats['manual_review'] / self.stats['processed']) * 100
        else:
            success_rate = 0
            review_rate = 0
        
        return {
            **self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'manual_review_rate': f"{review_rate:.1f}%"
        }
      //* Orchestrazione Pipeline RPA */
