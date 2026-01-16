# preprocessing/image_processor.py
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import structlog

logger = structlog.get_logger()

class ImagePreprocessor:
    """Preprocessing avanzato immagini per migliorare OCR"""
    
    def __init__(self):
        self.logger = logger.bind(component="ImagePreprocessor")
    
    def preprocess_invoice(
        self, 
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Pipeline completa di preprocessing per fatture
        
        Args:
            image_path: Path all'immagine input
            output_path: Path opzionale per salvare l'immagine processata
            
        Returns:
            Immagine preprocessata come numpy array
        """
        self.logger.info("Inizio preprocessing", image=str(image_path))
        
        # Carica immagine
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossibile caricare immagine: {image_path}")
        
        # Pipeline di preprocessing
        img = self._resize_if_needed(img)
        img = self._convert_to_grayscale(img)
        img = self._denoise(img)
        img = self._deskew(img)
        img = self._binarize(img)
        img = self._remove_borders(img)
        img = self._enhance_contrast(img)
        
        # Salva se richiesto
        if output_path:
            cv2.imwrite(str(output_path), img)
            self.logger.info("Immagine processata salvata", output=str(output_path))
        
        return img
    
    def _resize_if_needed(self, img: np.ndarray, max_width: int = 3000) -> np.ndarray:
        """Ridimensiona immagine se troppo grande"""
        height, width = img.shape[:2]
        
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
            self.logger.debug("Immagine ridimensionata", 
                            original_width=width, 
                            new_width=max_width)
        
        return img
    
    def _convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Conversione in scala di grigi"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Riduzione rumore con filtro bilaterale"""
        # Bilateral filter preserva i bordi
        return cv2.bilateralFilter(img, 9, 75, 75)
    
    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Correzione inclinazione documento"""
        # Trova angolo di rotazione
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correggi angolo
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Applica rotazione solo se significativa (> 0.5 gradi)
        if abs(angle) > 0.5:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            self.logger.debug("Immagine deskewata", angle=f"{angle:.2f}°")
        
        return img
    
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Binarizzazione adattiva ottimizzata per testo"""
        # Otsu's binarization per trovare threshold ottimale
        _, binary = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Adaptive threshold per gestire illuminazione non uniforme
        adaptive = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Combina i due approcci (prendi il migliore)
        # Usa Otsu se contrast è buono, altrimenti adaptive
        contrast = img.std()
        if contrast > 50:
            return binary
        else:
            return adaptive
    
    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        """Rimuove bordi neri attorno al documento"""
        # Trova contorni
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return img
        
        # Trova il contorno più grande (dovrebbe essere il documento)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Aggiungi piccolo margine
        margin = 10
        y = max(0, y - margin)
        x = max(0, x - margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        
        return img[y:y+h, x:x+w]
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Migliora contrasto con CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def detect_document_regions(self, img: np.ndarray) -> dict:
        """
        Identifica regioni chiave del documento (header, body, footer)
        Utile per estrazione mirata
        """
        height, width = img.shape[:2]
        
        regions = {
            'header': img[0:int(height * 0.25), :],
            'body': img[int(height * 0.25):int(height * 0.75), :],
            'footer': img[int(height * 0.75):, :]
        }
        
        return regions
    
    def extract_tables(self, img: np.ndarray) -> list:
        """Estrae tabelle dal documento usando line detection"""
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
        )
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, vertical_kernel, iterations=2
        )
        
        # Combine
        table_mask = cv2.add(detect_horizontal, detect_vertical)
        
        # Find table contours
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small regions
            if w > 100 and h > 100:
                table_region = img[y:y+h, x:x+w]
                tables.append({
                    'image': table_region,
                    'bbox': (x, y, w, h)
                })
        
        return tables
      /* Image Preprocessing con OpenCV */
