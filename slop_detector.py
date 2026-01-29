"""
AI Text Detector Integration Module

This module integrates the slop-detector-bert model from Hugging Face
for detecting AI-generated text content. It can be used in combination
with the deepfake video detector for multi-modal analysis.

Model: gouwsxander/slop-detector-bert
- BERT-based classifier for detecting AI-generated text
- Trained on Wikipedia human-written vs AI-rewritten paragraphs
- Labels: LABEL_0 (HUMAN), LABEL_1 (AI)
- This is a PEFT/LoRA adapter on bert-base-cased
"""

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class SlopDetectionResult:
    """Result from AI text detection."""
    text: str
    label: str  # "HUMAN" or "AI"
    confidence: float
    is_ai_generated: bool


class SlopDetector:

    # Using the requested ModernBERT model
    # Note: ModernBERT requires transformers >= 4.48.0
    MODEL_NAME = "AICodexLab/answerdotai-ModernBERT-base-ai-detector"
    
    def __init__(self, device: Optional[str] = None):
       
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._loaded = False
        
    def load_model(self) -> None:
        """Lazily load the model from Hugging Face."""
        if self._loaded:
            return
            
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print(f"Loading ModernBERT detector on {self.device}...")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            
            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                num_labels=2,
                trust_remote_code=True 
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            
            self._loaded = True
            print("ModernBERT detector loaded successfully!")
            
        except Exception as e:
            print(f"Error loading ModernBERT detector: {e}")
            print("Tip: Ensure you have transformers>=4.48.0 installed.")
            raise
    
    def detect(self, text: str) -> SlopDetectionResult:
        
        self.load_model()
        
        if not text or not text.strip():
            return SlopDetectionResult(
                text=text,
                label="UNKNOWN",
                confidence=0.0,
                is_ai_generated=False
            )
        
        # Tokenize with truncation
        inputs = self._tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            
            # Get prediction
            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities[0, predicted_class_id].item() * 100
        
        # LABEL_1 = AI, LABEL_0 = HUMAN
        is_ai = predicted_class_id == 1
        label = "AI" if is_ai else "HUMAN"
        
        return SlopDetectionResult(
            text=text[:500] + "..." if len(text) > 500 else text,
            label=label,
            confidence=confidence,
            is_ai_generated=is_ai
        )
    
    def detect_batch(self, texts: list[str]) -> list[SlopDetectionResult]:
        self.load_model()
        
        results = []
        for text in texts:
            results.append(self.detect(text))
        
        return results
    
    def analyze_paragraphs(self, full_text: str) -> dict:
        self.load_model()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in full_text.split('\n') if len(p.strip()) > 20]
        
        if not paragraphs:
            return {
                "overall_label": "UNKNOWN",
                "overall_confidence": 0.0,
                "ai_probability": 0.0,
                "paragraph_count": 0,
                "ai_paragraph_count": 0,
                "details": []
            }
        
        # Analyze each paragraph
        paragraph_results = self.detect_batch(paragraphs)
        
        # Calculate aggregate metrics
        ai_count = sum(1 for r in paragraph_results if r.is_ai_generated)
        ai_confidences = [r.confidence for r in paragraph_results if r.is_ai_generated]
        human_confidences = [r.confidence for r in paragraph_results if not r.is_ai_generated]
        
        # Overall probability based on paragraph analysis
        ai_probability = (ai_count / len(paragraphs)) * 100
        
        # Determine overall label (majority vote with confidence weighting)
        if ai_count > len(paragraphs) / 2:
            overall_label = "AI"
            overall_confidence = sum(ai_confidences) / len(ai_confidences) if ai_confidences else 0
        else:
            overall_label = "HUMAN"
            overall_confidence = sum(human_confidences) / len(human_confidences) if human_confidences else 0
        
        return {
            "overall_label": overall_label,
            "overall_confidence": round(overall_confidence, 2),
            "ai_probability": round(ai_probability, 2),
            "paragraph_count": len(paragraphs),
            "ai_paragraph_count": ai_count,
            "details": [
                {
                    "paragraph_preview": r.text[:100] + "..." if len(r.text) > 100 else r.text,
                    "label": r.label,
                    "confidence": round(r.confidence, 2)
                }
                for r in paragraph_results
            ]
        }


# Singleton instance for easy import
_detector_instance: Optional[SlopDetector] = None


def get_slop_detector() -> SlopDetector:
    """Get or create the singleton SlopDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SlopDetector()
    return _detector_instance


def detect_ai_text(text: str) -> SlopDetectionResult:

    detector = get_slop_detector()
    return detector.detect(text)


def analyze_text_content(text: str) -> dict:

    detector = get_slop_detector()
    return detector.analyze_paragraphs(text)


# Example usage and testing
if __name__ == "__main__":
    # Test the detector
    test_texts = [
        # Human-like text (original Wikipedia style)
        "Born in Bristol and raised in Glastonbury to an English father and Belgian mother, "
        "Norris began competitive kart racing aged eight. After a successful karting career, "
        "which culminated in his victory at the direct-drive World Championship in 2014, "
        "Norris graduated to junior formulae.",
        
        # AI-like text (more polished/structured)
        "Born in Bristol and raised in Glastonbury to an English father and a Belgian mother, "
        "Norris began competing in karting at the age of eight. He enjoyed a successful karting "
        "career, culminating in his victory at the direct-drive World Championship in 2014, "
        "before progressing into the junior single-seater categories.",
    ]
    
    print("=" * 60)
    print("AI Text Detection Test")
    print("=" * 60)
    
    detector = SlopDetector()
    
    for i, text in enumerate(test_texts, 1):
        result = detector.detect(text)
        print(f"\nText {i}:")
        print(f"  Preview: {text[:80]}...")
        print(f"  Label: {result.label}")
        print(f"  Confidence: {result.confidence:.2f}%")
        print(f"  Is AI Generated: {result.is_ai_generated}")
