from transformers import pipeline
import logging

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


class FootwearRecommender:
    def __init__(self):
        """Initialize the Hugging Face text generation pipeline."""
        self.generator = pipeline(
            "text2text-generation", 
            model="google/flan-t5-small", 
            device=-1  # CPU only
        )
    
    def generate_recommendations(self, length_mm, men_size, women_size, width_category, arch_type, use_case, sizing_reference):
        """
        Generate shoe recommendations based on foot analysis.
        Always uses the comprehensive fallback system for reliable, detailed recommendations.
        """
        # Use the detailed fallback recommendations which are more reliable and comprehensive
        return self._get_fallback_recommendations(width_category, arch_type, use_case)
    
    def _get_fallback_recommendations(self, width_category, arch_type, use_case):
        """Provide conversational, user-friendly recommendations."""
        
        # Start with arch-specific advice
        arch_advice = ""
        if arch_type == 'flat':
            arch_advice = "Since you have flat feet, your arches tend to collapse inward when you walk or run. This means you'll want **stability or motion control shoes** that help prevent your foot from rolling too far inward. Look for shoes with firm support built into the inner edge of the sole."
        elif arch_type == 'high':
            arch_advice = "With high arches, your feet don't absorb impact as naturally as flatter feet do. You'll be most comfortable in **neutral cushioned shoes** with plenty of soft padding. Avoid shoes with rigid arch support - they might actually feel uncomfortable since your natural arch is already doing the work."
        else:  # normal arch
            arch_advice = "Good news! Your normal arch means most shoe types will work well for you. You can choose either **neutral or stability shoes** based on your comfort preference, and you'll likely be happy with either option."
        
        # Add width-specific advice
        width_advice = ""
        if width_category == 'wide':
            width_advice = "Your foot is on the wider side, so regular width shoes might feel tight and pinch your toes. When shopping, look for shoes marked **'W' or '2E' for wide width**. Make sure there's plenty of room in the toe area - you should be able to wiggle your toes comfortably."
        elif width_category == 'narrow':
            width_advice = "Since your foot is narrower than average, you'll want to look for **narrow width shoes** (marked 'N' or 'B' on the box). This helps ensure your heel doesn't slip around and your foot stays secure. Pay special attention to how the shoe grips your heel and midfoot."
        else:  # regular width
            width_advice = "Your foot width is pretty standard, so regular width shoes should fit you well without any special considerations needed."
        
        # Add use case advice
        activity_advice = ""
        if use_case == 'Running':
            activity_advice = "For running, prioritize shoes with good cushioning to protect your feet from repeated impact. A good rule of thumb is to replace your running shoes every 300-500 miles - worn out shoes lose their support and can lead to discomfort or injury."
        elif use_case == 'Walking':
            activity_advice = "For walking, comfort is key since you'll likely be wearing them for extended periods. Look for shoes with good all-day comfort and a lower heel drop (the difference between heel and toe height) for a more natural walking motion."
        elif use_case == 'Hiking':
            activity_advice = "For hiking, you'll want either hiking boots or trail shoes with good grip and ankle support. If you often hike in wet conditions, waterproof options can keep your feet dry and comfortable."
        elif use_case == 'Court Sports':
            activity_advice = "Court sports require shoes that can handle quick direction changes and lateral movements. Look for court-specific shoes with good ankle support to help prevent injuries during play."
        elif use_case == 'Cleats/Boots':
            activity_advice = "For cleats or specialized boots, proper fit is absolutely crucial for both performance and injury prevention. Make sure you get the right type for your specific sport and playing surface."
        
        # Combine all advice into a conversational paragraph
        recommendation_text = f"{arch_advice}\n\n{width_advice}"
        if activity_advice:
            recommendation_text += f"\n\n{activity_advice}"
            
        recommendation_text += f"\n\n**Quick shopping tip:** When trying on shoes, do it later in the day when your feet are slightly swollen (like they'll be during activity). This ensures you get the most comfortable fit!"
        
        return recommendation_text


def get_recommendations(length_mm, men_size, women_size, width_category, arch_type, use_case, sizing_reference):
    """
    Convenience function to get recommendations.
    Creates a new recommender instance and generates recommendations.
    """
    recommender = FootwearRecommender()
    return recommender.generate_recommendations(
        length_mm, men_size, women_size, width_category, arch_type, use_case, sizing_reference
    )