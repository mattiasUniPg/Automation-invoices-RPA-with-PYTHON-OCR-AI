# ocr/ocr_engine.py
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import structlog

from config import settings

logger = structlog.get_logger()

@dataclass
class OCRResult:
    """Risultato estrazione OCR"""
    text: str
    confidence: float
    word_data: List[Dict]
    layout_data: Optional[Dict] = None

@dataclass
class InvoiceField:
    """Campo estratto da fattura"""
    value: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    field_type: str

class InvoiceOCREngine:
    """Engine OCR specializzato per fatture"""
    
    def __init__(self):
        self.logger = logger.bind(component="OCREngine")
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
        
        # Pattern regex per estrazione dati
        self.patterns = {
            'invoice_number': [
                r'(?:fattura|invoice|n[°\.º]?)\s*[:\-]?\s*(\d{4,}[/\-]?\d*)',
                r'(?:FT|INV|DOC)[:\-\s]*(\d{4,})',
                r'numero\s+(?:fattura|documento)[:\s]+(\d+)',
            ],
            'date': [
                r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
            ],
            'vat_number': [
                r'(?:p\.?\s*iva|partita\s+iva|vat)[:\s]*(\d{11})',
                r'(?:tax\s+id|fiscal\s+code)[:\s]*(\d{11})',
            ],
            'amount': [
                r'(?:totale|total|importo)[:\s]+€?\s*([\d\.,]+)',
                r'(?:grand\s+total|net\s+amount)[:\s]+€?\s*([\d\.,]+)',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ]
        }
    
    def extract_text(
        self, 
        image: np.ndarray,
        config: str = '--oem 3 --psm 6'
    ) -> OCRResult:
        """
        Estrae testo da immagine con dati dettagliati
        
        Args:
            image: Immagine preprocessata
            config: Configurazione Tesseract
            
        Returns:
            OCRResult con testo e metadati
        """
        self.logger.info("Inizio estrazione OCR")
        
        # Estrazione con layout data
        data = pytesseract.image_to_data(
            image,
            lang=settings.TESSERACT_LANG,
            config=config,
            output_type=Output.DICT
        )
        
        # Estrazione testo completo
        text = pytesseract.image_to_string(
            image,
            lang=settings.TESSERACT_LANG,
            config=config
        )
        
        # Filtra parole con confidence bassa
        word_data = []
        confidences = []
        
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            if conf > 0:
                word_data.append({
                    'text': data['text'][i],
                    'confidence': conf,
                    'bbox': (
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    ),
                    'block_num': data['block_num'][i],
                    'line_num': data['line_num'][i]
                })
                confidences.append(conf)
        
        avg_confidence = np.mean(confidences) if confidences else 0
        
        self.logger.info(
            "OCR completato",
            words_extracted=len(word_data),
            avg_confidence=f"{avg_confidence:.2f}%"
        )
        
        return OCRResult(
            text=text,
            confidence=avg_confidence,
            word_data=word_data,
            layout_data=data
        )
    
    def extract_invoice_fields(
        self, 
        text: str,
        word_data: List[Dict]
    ) -> Dict[str, InvoiceField]:
        """
        Estrae campi strutturati da testo OCR
        
        Args:
            text: Testo estratto
            word_data: Dati word-level da OCR
            
        Returns:
            Dizionario con campi estratti
        """
        fields = {}
        
        # Estrai ogni tipo di campo
        for field_type, patterns in self.patterns.items():
            field = self._extract_field(text, patterns, word_data, field_type)
            if field:
                fields[field_type] = field
        
        # Post-processing specifico
        fields = self._post_process_fields(fields)
        
        return fields
    
    def _extract_field(
        self,
        text: str,
        patterns: List[str],
        word_data: List[Dict],
        field_type: str
    ) -> Optional[InvoiceField]:
        """Estrae singolo campo usando pattern regex"""
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                
                # Trova bbox approssimativo dal word_data
                bbox = self._find_bbox_for_text(value, word_data)
                
                # Calcola confidence media per questo campo
                confidence = self._calculate_field_confidence(value, word_data)
                
                return InvoiceField(
                    value=value.strip(),
                    confidence=confidence,
                    bbox=bbox,
                    field_type=field_type
                )
        
        return None
    
    def _find_bbox_for_text(
        self, 
        text: str, 
        word_data: List[Dict]
    ) -> Tuple[int, int, int, int]:
        """Trova bounding box per testo specifico"""
        
        # Cerca corrispondenza parziale nelle parole estratte
        for word in word_data:
            if text.lower() in word['text'].lower():
                return word['bbox']
        
        # Default bbox se non trovato
        return (0, 0, 0, 0)
    
    def _calculate_field_confidence(
        self, 
        value: str, 
        word_data: List[Dict]
    ) -> float:
        """Calcola confidence per un campo specifico"""
        
        # Trova parole che matchano il valore
        matching_words = [
            w for w in word_data 
            if any(part.lower() in w['text'].lower() for part in value.split())
        ]
        
        if matching_words:
            return np.mean([w['confidence'] for w in matching_words])
        
        return 0.0
    
    def _post_process_fields(
        self, 
        fields: Dict[str, InvoiceField]
    ) -> Dict[str, InvoiceField]:
        """Post-processing e normalizzazione campi"""
        
        # Normalizza P.IVA
        if 'vat_number' in fields:
            vat = fields['vat_number'].value
            vat = re.sub(r'[^\d]', '', vat)  # Solo numeri
            if len(vat) == 11:
                fields['vat_number'].value = vat
            else:
                self.logger.warning("P.IVA non valida", vat=vat)
        
        # Normalizza date
        if 'date' in fields:
            date_str = fields['date'].value
            date_normalized = self._normalize_date(date_str)
            if date_normalized:
                fields['date'].value = date_normalized
        
        # Normalizza importi
        if 'amount' in fields:
            amount_str = fields['amount'].value
            # Rimuovi separatori migliaia e converti decimali
            amount_str = amount_str.replace('.', '').replace(',', '.')
            try:
                amount_float = float(amount_str)
                fields['amount'].value = f"{amount_float:.2f}"
            except ValueError:
                self.logger.warning("Importo non valido", amount=amount_str)
        
        return fields
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalizza data in formato ISO"""
        from datetime import datetime
        
        # Prova vari formati
        formats = [
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%d.%m.%Y',
            '%Y-%m-%d',
            '%d/%m/%y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def extract_line_items(
        self, 
        image: np.ndarray,
        table_regions: List[Dict]
    ) -> List[Dict]:
        """Estrae righe di dettaglio da tabelle"""
        
        line_items = []
        
        for table in table_regions:
            table_img = table['image']
            
            # OCR sulla tabella
            table_data = pytesseract.image_to_data(
                table_img,
                lang=settings.TESSERACT_LANG,
                output_type=Output.DICT
            )
            
            # Raggruppa per righe
            lines = {}
            for i in range(len(table_data['text'])):
                if int(table_data['conf'][i]) > 30:
                    line_num = table_data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []
                    lines[line_num].append(table_data['text'][i])
            
            # Estrai items dalle righe
            for line_texts in lines.values():
                line_text = ' '.join(line_texts)
                
                # Pattern per righe prodotto: descrizione, quantità, prezzo
                item_pattern = r'(.+?)\s+(\d+)\s+€?\s*([\d\.,]+)'
                match = re.search(item_pattern, line_text)
                
                if match:
                    line_items.append({
                        'description': match.group(1).strip(),
                        'quantity': int(match.group(2)),
                        'price': float(match.group(3).replace(',', '.'))
                    })
        
        return line_items
    
    def visualize_ocr_results(
        self, 
        image: np.ndarray,
        ocr_result: OCRResult,
        output_path: Path
    ) -> None:
        """Crea visualizzazione OCR con bounding boxes"""
        
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Disegna bbox per ogni parola
        for word in ocr_result.word_data:
            x, y, w, h = word['bbox']
            conf = word['confidence']
            
            # Colore basato su confidence
            if conf > 80:
                color = (0, 255, 0)  # Verde
            elif conf > 60:
                color = (0, 255, 255)  # Giallo
            else:
                color = (0, 0, 255)  # Rosso
            
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis_img,
                f"{conf}%",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        cv2.imwrite(str(output_path), vis_img)
     //*  OCR Engine Avanzato */
