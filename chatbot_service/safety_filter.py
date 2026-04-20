"""
Safety filter to block dangerous medical questions - BALANCED VERSION
"""
import re
from typing import Tuple


class SafetyFilter:
    """Filter to ensure chatbot only handles educational questions"""
    
    # Blocked keywords/patterns (medical diagnosis, prescriptions, emergencies)
    BLOCKED_PATTERNS = [
        # Diagnosis confirmation requests (very specific)
        r'ده\s+(ADHD|توحد|عسر قراءة)\s+أكيد',  # "This is ADHD for sure?"
        r'أكيد\s+(ADHD|توحد|عسر قراءة)',  # "Definitely ADHD?"
        r'يعني\s+(عنده|مريض|لديه)\s+(ADHD|توحد|عسر قراءة)',  # "Means he has ADHD?"
        r'تشخيص\s+(ADHD|توحد|عسر قراءة)',  # "Diagnose ADHD"
        
        # Medication/Drug requests (very specific)
        r'دواء|أدوية',  # Medicine/drugs
        r'علاج دوائي',  # Drug treatment  
        r'ريتالين|كونسيرتا|ستراتيرا|أديرال',  # Common ADHD meds
        r'جرعة',  # Dosage
        r'وصفة طبية',  # Prescription
        r'أحسن دوا|أفضل دواء|أحسن علاج دوائي',  # "Best medicine"
        r'ينفع أديه\s+دوا|أعطيه\s+دواء',  # "Can I give him medicine"
        r'أبدأ\s+دواء|أوقف\s+دواء',  # "Start/stop medicine"
        
        # Treatment plans
        r'خطة علاج\s+دوائي|علاج كامل',  # Treatment plan
        r'بديل الطبيب|بدلاً من الطبيب',
        
        # English equivalents
        r'\bdiagnos(e|is)\b',
        r'\b(drug|medication|prescription)\b',
        r'\b(ritalin|concerta|strattera|adderall)\b',
        r'\bdosage\b',
        r'replace.*doctor|instead.*of.*doctor',
        
        # Emergency situations  
        r'طوارئ|حالة طارئة',
        r'emergency',
        r'حادث خطير',
    ]
    
    # Educational/allowed patterns (still safe)
    ALLOWED_PATTERNS = [
        r'تمارين|أنشطة',
        r'exercise|activity',
        r'استراتيجية|نصيحة',
        r'strategy|advice|tip',
        r'فهم|تعلم|اشرح',
        r'understand|learn|explain',
        r'دعم عاطفي',
        r'emotional support',
        r'جدول يومي|روتين',
        r'daily schedule|routine',
    ]
    
    def __init__(self):
        """Initialize compiled regex patterns"""
        self.blocked_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.BLOCKED_PATTERNS]
        self.allowed_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.ALLOWED_PATTERNS]
    
    def is_safe(self, question: str) -> Tuple[bool, str]:
        """
        Check if question is safe (educational) or blocked (medical)
        
        Args:
            question: User's question text
            
        Returns:
            Tuple of (is_safe: bool, reason: str)
        """
        # Check for blocked patterns
        for pattern in self.blocked_regex:
            if pattern.search(question):
                return False, "medical_advice_blocked"
        
        # If no blocked patterns, it's safe
        return True, "educational_question"
    
    def get_blocked_response(self) -> str:
        """Get standard response for blocked questions"""
        return """⚠️ عذراً، لا أستطيع تقديم مشورة طبية أو تشخيصات

أنا مساعد تعليمي مصمم لدعم الآباء بمعلومات عامة فقط.

🏥 **من فضلك استشر:**
- طبيب أطفال متخصص
- أخصائي نفسي للأطفال
- مركز صحي متخصص

يمكنني مساعدتك في:
✅ فهم الأعراض والسلوكيات بشكل عام
✅ اقتراح أنشطة وتمارين تعليمية
✅ استراتيجيات الدعم العاطفي
✅ نصائح تربوية عامة"""


# Global instance
safety_filter = SafetyFilter()
