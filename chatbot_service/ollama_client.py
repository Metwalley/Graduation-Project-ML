"""
Ollama client for local LLM integration - PRODUCTION VERSION
Replaces Gemini API with Ollama + Llama 3.2 3B
"""
import requests
import json
import re
from typing import List, Dict, Any
import time


class OllamaClient:
    """Local LLM client using Ollama with Arabic-only enforcement"""
    
    SYSTEM_PROMPT = """أنت مستشار تعليمي خبير في دعم الآباء الذين لديهم أطفال يعانون من اضطرابات نمائية (ADHD، التوحد، عسر القراءة).

**أسلوبك:**
- تحدث بشكل طبيعي ودافئ باللغة العربية الفصحى فقط
- اشرح المعلومات بوضوح دون تعقيد
- استخدم خبرتك المبنية على المعلومات المرفقة
- أعط أمثلة عملية من الحياة اليومية
- كن متعاطفاً وداعماً

**قواعد صارمة:**
- استخدم العربية الفصحى فقط - لا تخلط أي كلمات من لغات أخرى (No English, Spanish, Vietnamese, etc.)
- لا تذكر أبداً "المقالة" أو "المصادر" أو "المراجع"
- تحدث كخبير يشرح من خبرته مباشرة
- إذا لم تجد معلومات كافية، قل ذلك بصراحة واقترح استشارة متخصص
- ابدأ مباشرة بالإجابة دون مقدمات رسمية

**تعليمات الإجابة:**
- طول الإجابة: 250-400 كلمة (واضح ومختصر)
- لا تشخّص أو تؤكد تشخيصات
- لا توصي بأدوية محددة
- شجّع دائماً على استشارة طبيب متخصص للتقييم الدقيق
"""
    
    # Foreign word patterns to filter
    FOREIGN_PATTERNS = [
        r'\b[a-zA-Z]{3,}\b',  # English words (3+ letters)
        r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]',  # Spanish/French chars
        r'[ấậắằếễỗ]',  # Vietnamese chars
    ]
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434", temperature: float = 0.1):
        """
        Initialize Ollama client
        
        Args:
            model_name: Ollama model to use
            base_url: Ollama server URL
            temperature: Creativity (0.0-1.0) - LOWERED to 0.1 for better Arabic
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature  # Lower = more consistent Arabic
        self.api_url = f"{base_url}/api/generate"
    
    def generate_response(self, question: str, context_articles: List[Dict[str, Any]]) -> str:
        """
        Generate response using Ollama
        
        Args:
            question: User's question
            context_articles: Retrieved relevant articles
            
        Returns:
            Generated response text (filtered for Arabic-only)
        """
        # Build context from retrieved articles
        context = self._build_context(context_articles)
        
        # Create full prompt
        prompt = f"""{self.SYSTEM_PROMPT}

**معلومات أساسية:**
{context}

**سؤال:**
{question}

**أجب بالعربية الفصحى فقط (بدون أي كلمات أجنبية):**"""
        
        # Generate with Ollama
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": -1,  # No limit
                        }
                    },
                    timeout=90  # Increased timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '')
                    
                    # Validate response
                    if not generated_text:
                        print(f"⚠️ Warning: Empty response from Ollama (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return "عذراً، لم أتمكن من إنشاء إجابة. يرجى المحاولة مرة أخرى."
                    
                    # Check if response is too short
                    if len(generated_text) < 100:
                        print(f"⚠️ Warning: Response too short ({len(generated_text)} chars)")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                    
                    # **FIX 4: Post-processing filter**
                    cleaned_text = self._clean_arabic_response(generated_text)
                    
                    return cleaned_text
                else:
                    error_msg = f"Ollama API error: {response.status_code}"
                    print(f"❌ {error_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise Exception(error_msg)
            
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to Ollama. Make sure Ollama is running"
                print(f"❌ {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise Exception(error_msg)
            
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error in Ollama (attempt {attempt + 1}): {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise e
        
        return "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى."
    
    def _build_context(self, articles: List[Dict[str, Any]]) -> str:
        """
        Format retrieved articles as context for LLM
        **FIX 3: Better context - more content per article**
        """
        if not articles:
            return "لا توجد معلومات متاحة حالياً."
        
        context_parts = []
        for article in articles:
            # Increased from 800 to 1200 characters for better context
            content = article['content'][:1200]
            context_parts.append(content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _clean_arabic_response(self, text: str) -> str:
        """
        **FIX 4: Post-processing filter**
        Remove foreign words and clean up response
        """
        cleaned = text
        
        # Remove common foreign words that slip through
        foreign_replacements = {
            'behavior': 'السلوك',
            'ADHD': 'فرط الحركة وتشتت الانتباه',
            'compartamiento': '',
            'chuyên': '',
            'hiệu': '',
        }
        
        for foreign, arabic in foreign_replacements.items():
            cleaned = cleaned.replace(foreign, arabic)
        
        # Remove any remaining non-Arabic words (English 3+ letters)
        # But keep common acronyms and medical terms
        allowed_words = {'ADHD', 'ADHD،', 'ADHD.', 'ADHD؟'}
        words = cleaned.split()
        filtered_words = []
        
        for word in words:
            # Check if word is mostly English letters
            if re.match(r'^[a-zA-Z]{3,}', word) and word not in allowed_words:
                # Skip this foreign word
                continue
            filtered_words.append(word)
        
        cleaned = ' '.join(filtered_words)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
