import re
from typing import Optional

def generate_presentation_filename(content: str, file_extension: str = "pptx") -> str:
    """
    Generate a relevant filename based on presentation content.
    
    Args:
        content: The original presentation prompt/content
        file_extension: File extension (pptx, pdf)
    
    Returns:
        A relevant filename in format: "{topic}'s {document_type}"
    """
    # Clean and extract the main topic from content
    cleaned_content = content.strip().lower()
    
    # Try to identify if this is about a person, company, or topic
    if _is_about_person(cleaned_content):
        base_name = _extract_person_name(cleaned_content)
        document_type = "Biography"
    elif _is_about_company(cleaned_content):
        base_name = _extract_company_name(cleaned_content)
        document_type = "Overview"
    elif _is_about_journey(cleaned_content):
        base_name = _extract_topic(cleaned_content)
        document_type = "Journey"
    elif _is_about_analysis(cleaned_content):
        base_name = _extract_topic(cleaned_content)
        document_type = "Analysis"
    elif _is_about_presentation(cleaned_content):
        base_name = _extract_topic(cleaned_content)
        document_type = "Presentation"
    else:
        # Default: extract first meaningful phrase
        base_name = _extract_topic(cleaned_content)
        document_type = "Presentation"
    
    # Sanitize the filename
    safe_name = _sanitize_filename(f"{base_name}'s {document_type}")
    
    return f"{safe_name}.{file_extension}"

def _is_about_person(content: str) -> bool:
    """Check if content is about a specific person"""
    person_indicators = [
        r'\b(biography|about|life\s+story|story\s+of|journey\s+of)\b',
        r'\b(donald\s+trump|joe\s+biden|elon\s+musk|steve\s+jobs|bill\s+gates)\b',
        r'\b\w+\s+\w+\b(?:\'s)?\s+(?:biography|life|story)'
    ]
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in person_indicators)

def _is_about_company(content: str) -> bool:
    """Check if content is about a company"""
    company_indicators = [
        r'\b(company|business|organization|corporation)\b',
        r'\b(overview|analysis|profile)\s+of\s+\w+\s+(?:inc|llc|corp|ltd)\b',
        r'\bapple\s+inc|microsoft|google|amazon|tesla|meta\b'
    ]
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in company_indicators)

def _is_about_journey(content: str) -> bool:
    """Check if content describes a journey or progression"""
    journey_indicators = [
        r'\b(journey|path|road|story|progression|evolution|development)\b',
        r'\bfrom\s+\w+\s+to\s+\w+\b',
        r'\b\w+\'s\s+journey\b'
    ]
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in journey_indicators)

def _is_about_analysis(content: str) -> bool:
    """Check if content is an analysis"""
    analysis_indicators = [
        r'\b(analysis|review|critique|evaluation|assessment)\b',
        r'\b(analyzing|examining|breaking\s+down|deep\s+dive)\b'
    ]
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in analysis_indicators)

def _is_about_presentation(content: str) -> bool:
    """Check if content is explicitly about creating a presentation"""
    presentation_indicators = [
        r'\bpresentation\s+(?:about|on|regarding)\b',
        r'\bcreate\s+(?:a\s+)?presentation\s+about\b',
        r'\b(?:a\s+)?presentation\s+(?:on|about)\s+\w+'
    ]
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in presentation_indicators)

def _extract_person_name(content: str) -> str:
    """Extract person's name from content"""
    # Try to match "about [Name]" or "[Name]'s story" patterns
    patterns = [
        r'presentation\s+about\s+([a-zA-Z\s]+)',
        r'([a-zA-Z\s]+)\'s\s+(?:biography|life|story|journey)',
        r'(?:biography|life\s+story|story\s+of)\s+([a-zA-Z\s]+)',
        r'donald\s+trump|joe\s+biden|elon\s+musk|steve\s+jobs|bill\s+gates'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            name = match.group(1) if match.groups() else match.group(0)
            return name.strip().title()
    
    # Fallback: extract capitalized words
    capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    return capitalized_words[0] if capitalized_words else "Unknown Person"

def _extract_company_name(content: str) -> str:
    """Extract company name from content"""
    patterns = [
        r'overview\s+of\s+([a-zA-Z\s]+)',
        r'([a-zA-Z\s]+)(?:\s+inc|\s+llc|\s+corp|\s+ltd)',
        r'apple\s+inc|microsoft|google|amazon|tesla|meta'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            name = match.group(1) if match.groups() else match.group(0)
            return name.strip().title()
    
    return "Unknown Company"

def _extract_topic(content: str) -> str:
    """Extract main topic from content"""
    # Remove common presentation prefixes
    cleaned = re.sub(r'^(?:create\s+|make\s+|generate\s+)?a?\s*presentation\s+(?:about|on|regarding)\s+', '', content, flags=re.IGNORECASE)
    
    # Look for the first substantial phrase (3+ words)
    substantial_phrases = re.findall(r'\b([a-zA-Z]+\s+[a-zA-Z]+\s+(?:[a-zA-Z]+\s*)?)', cleaned)
    
    for phrase in substantial_phrases:
        if len(phrase.strip().split()) >= 2:  # At least 2 words
            return phrase.strip().title()
    
    # Fallback: first sentence or key phrase
    sentences = re.split(r'[.!?]+', cleaned)
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence.split()) >= 2:
            return first_sentence.title()
    
    # Final fallback: extract key terms
    key_terms = re.findall(r'\b([a-zA-Z]{4,})\b', content)
    if key_terms:
        return key_terms[0].title()
    
    return "Unknown Topic"

def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem compatibility"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100].strip()
    return sanitized.strip()