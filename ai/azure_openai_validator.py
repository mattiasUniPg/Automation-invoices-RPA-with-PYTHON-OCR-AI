# ai/azure_openai_validator.py
from openai import AzureOpenAI
from typing import Dict, List, Optional
import json
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, field_validator

from config import settings

logger = structlog.get_logger()

class InvoiceData(BaseModel):
    """Modello Pydantic per dati fattura validati"""
    
    invoice_number: str = Field(..., description="Numero fattura")
    invoice_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    
    supplier_name: str
    supplier_vat: str = Field(..., pattern=r'^\d{11}$')
    supplier_address: Optional[str] = None
    
    customer_name: str
    customer_vat: str = Field(..., pattern=r'^\d{11}$')
    customer_address: Optional[str] = None
    
    subtotal: float = Field(..., gt=0)
    vat_rate: float = Field(default=0.22, ge=0, le=1)
    vat_amount: float = Field(..., ge=0)
    total_amount: float = Field(..., gt=0)
    
    line_items: List[Dict] = Field(default_factory=list)
    payment_terms: Optional[str] = None
    due_date: Optional[str] = None
    
    currency: str = Field(default="EUR", pattern=r'^[A-Z]{3}$')
    
    # Metadati validazione
    ocr_confidence: float = Field(..., ge=0, le=100)
    ai_validation_score: float = Field(..., ge=0, le=1)
    validation_notes: List[str] = Field(default_factory=list)
    requires_manual_review: bool = False
    
    @field_validator('total_amount')
    @classmethod
    def validate_total(cls, v: float, info) -> float:
        """Valida che total = subtotal + vat"""
        if 'subtotal' in info.data and 'vat_amount' in info.data:
            expected_total = info.data['subtotal'] + info.data['vat_amount']
            if abs(v - expected_total) > 0.01:
                raise ValueError(
                    f"Total {v} doesn't match subtotal + VAT {expected_total}"
                )
        return v
    
    @field_validator('vat_amount')
    @classmethod
    def validate_vat(cls, v: float, info) -> float:
        """Valida calcolo IVA"""
        if 'subtotal' in info.data and 'vat_rate' in info.data:
            expected_vat = info.data['subtotal'] * info.data['vat_rate']
            if abs(v - expected_vat) > 0.01:
                raise ValueError(
                    f"VAT {v} doesn't match subtotal * rate {expected_vat}"
                )
        return v

class AzureOpenAIValidator:
    """Validatore AI per dati fattura estratti via OCR"""
    
    def __init__(self):
        self.logger = logger.bind(component="AIValidator")
        
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def validate_and_structure_invoice(
        self,
        ocr_text: str,
        extracted_fields: Dict,
        ocr_confidence: float
    ) -> InvoiceData:
        """
        Valida e struttura dati fattura usando Azure OpenAI
        
        Args:
            ocr_text: Testo completo estratto da OCR
            extracted_fields: Campi estratti via regex
            ocr_confidence: Confidence media OCR
            
        Returns:
            InvoiceData validato e strutturato
        """
        self.logger.info("Inizio validazione AI")
        
        # Prompt engineering per validazione
        system_prompt = self._create_validation_prompt()
        user_message = self._format_extraction_data(
            ocr_text, 
            extracted_fields
        )
        
        # Chiamata Azure OpenAI
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # Bassa per output deterministico
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Parse risposta
        ai_response = response.choices[0].message.content
        validated_data = json.loads(ai_response)
        
        self.logger.info(
            "Validazione AI completata",
            tokens_used=response.usage.total_tokens
        )
        
        # Aggiungi metadati
        validated_data['ocr_confidence'] = ocr_confidence
        validated_data['ai_validation_score'] = validated_data.get(
            'confidence_score', 0.9
        )
        
        # Validazione con Pydantic
        try:
            invoice_data = InvoiceData.model_validate(validated_data)
        except Exception as e:
            self.logger.error("Validazione Pydantic fallita", error=str(e))
            # Flagga per revisione manuale
            validated_data['requires_manual_review'] = True
            validated_data['validation_notes'] = [str(e)]
            invoice_data = InvoiceData.model_validate(
                validated_data,
                strict=False
            )
        
        # Business rules validation
        invoice_data = self._apply_business_rules(invoice_data)
        
        return invoice_data
    
    def _create_validation_prompt(self) -> str:
        """Crea system prompt per validazione"""
        
        return f"""Sei un esperto contabile specializzato nell'analisi e validazione di fatture italiane.

Il tuo compito è:
1. Analizzare il testo OCR estratto da una fattura
2. Validare e correggere i dati estratti automaticamente
3. Identificare inconsistenze o errori
4. Restituire dati strutturati in formato JSON

REGOLE DI VALIDAZIONE:

**Numero Fattura:**
- Deve essere presente e univoco
- Formato tipico: numero progressivo con anno (es. 2024/001, FT-2024-123)

**Date:**
- Formato ISO: YYYY-MM-DD
- Data fattura deve essere valida e non futura
- Data scadenza deve essere successiva alla data fattura

**Partite IVA:**
- Esattamente 11 cifre
- Solo numeri, rimuovi spazi e caratteri speciali
- Valida checksum se possibile

**Importi:**
- Imponibile (subtotal): importo pre-IVA
- IVA (vat_amount): deve essere = subtotal * vat_rate
- Totale (total_amount): deve essere = subtotal + vat_amount
- Aliquota IVA standard italiana: {settings.VAT_RATE} (22%)
- Tolleranza arrotondamenti: ±0.01€

**Coerenza:**
- Tutti gli importi devono essere coerenti matematicamente
- Se trovi discrepanze, segnalale in validation_notes

**Output JSON richiesto:**
{{
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "supplier_name": "string",
  "supplier_vat": "11 cifre",
  "customer_name": "string",
  "customer_vat": "11 cifre",
  "subtotal": float,
  "vat_rate": float (default 0.22),
  "vat_amount": float,
  "total_amount": float,
  "line_items": [
    {{"description": "string", "quantity": int, "unit_price": float, "total": float}}
  ],
  "payment_terms": "string (es. 'Pagamento a 30 giorni')",
  "due_date": "YYYY-MM-DD" (opzionale),
  "currency": "EUR",
  "confidence_score": float (0-1, tua confidenza nella validazione),
  "validation_notes": ["lista di note o correzioni applicate"],
  "requires_manual_review": boolean (true se ci sono dubbi significativi)
}}

IMPORTANTE:
- Se mancano dati critici, usa null
- Se dati sembrano errati ma non sei sicuro, segnala in validation_notes
- Se confidence_score < 0.7, imposta requires_manual_review = true
"""
    
    def _format_extraction_data(
        self, 
        ocr_text: str, 
        fields: Dict
    ) -> str:
        """Formatta dati per prompt"""
        
        return f"""Valida questa fattura estratta via OCR.

TESTO COMPLETO OCR:
{ocr_text}

CAMPI ESTRATTI AUTOMATICAMENTE:
{json.dumps(fields, indent=2, ensure_ascii=False)}

Analizza il testo, valida i campi estratti, correggi eventuali errori e restituisci il JSON validato.
"""
    
    def _apply_business_rules(self, invoice: InvoiceData) -> InvoiceData:
        """Applica regole business"""
        
        # Verifica soglia auto-approvazione
        if invoice.total_amount > settings.AUTO_APPROVE_THRESHOLD:
            invoice.requires_manual_review = True
            invoice.validation_notes.append(
                f"Importo €{invoice.total_amount:.2f} supera soglia auto-approvazione"
            )
        
        # Verifica importo massimo
        if invoice.total_amount > settings.MAX_INVOICE_AMOUNT:
            invoice.requires_manual_review = True
            invoice.validation_notes.append(
                f"Importo €{invoice.total_amount:.2f} supera limite massimo"
            )
        
        # Verifica confidence OCR
        if invoice.ocr_confidence < settings.OCR_CONFIDENCE_THRESHOLD:
            invoice.requires_manual_review = True
            invoice.validation_notes.append(
                f"Confidence OCR bassa: {invoice.ocr_confidence:.1f}%"
            )
        
        # Verifica confidence AI
        if invoice.ai_validation_score < 0.7:
            invoice.requires_manual_review = True
            invoice.validation_notes.append(
                f"Confidence AI bassa: {invoice.ai_validation_score:.2f}"
            )
        
        return invoice
    
    def semantic_similarity_check(
        self,
        ocr_text: str,
        validated_data: InvoiceData
    ) -> float:
        """Verifica coerenza semantica tra OCR e dati validati"""
        
        # Usa embedding per similarity check
        prompt = f"""Confronta il testo OCR originale con i dati estratti e valuta la coerenza.

TESTO ORIGINALE:
{ocr_text[:1000]}

DATI ESTRATTI:
Numero: {validated_data.invoice_number}
Data: {validated_data.invoice_date}
Fornitore: {validated_data.supplier_name}
Cliente: {validated_data.customer_name}
Totale: €{validated_data.total_amount}

Rispondi con un punteggio di coerenza da 0 a 1, dove:
- 1.0 = dati perfettamente coerenti con il testo
- 0.5 = alcune discrepanze minori
- 0.0 = dati completamente incoerenti

Restituisci SOLO il numero (es. 0.85)"""
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        try:
            similarity_score = float(response.choices[0].message.content.strip())
            return min(max(similarity_score, 0.0), 1.0)
        except:
            return 0.5  # Default se parsing fallisce
//* Validazione con Azure OpenAI */
