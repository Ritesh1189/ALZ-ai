"""
Module: text_cleaner.py
Responsible for cleaning, normalizing resume text, and segmenting resume sections
while carefully preserving critical technical symbols such as C++, C#, and .NET.
"""

import re
from typing import Dict, List, Optional


TOKEN_PROTECTION_MAP = {
    r"(?:(?<=[\s,;(\[])|^)c\+\+(?=[,\s;.)\]]|$)": "__token_cpp__",
    r"(?:(?<=[\s,;(\[])|^)c\#(?=[,\s;.)\]]|$)": "__token_csharp__",
    r"(?:(?<=[\s,;(\[])|^)\.net(?=[,\s;.)\]]|$)": "__token_dotnet__",
    r"\bnode\.js\b": "__token_nodejs__",
    r"\bvue\.js\b": "__token_vuejs__",
    r"\breact\.js\b": "__token_reactjs__",
    r"\bci\/cd\b": "__token_cicd__",
    r"\bscikit-learn\b": "__token_sklearn__",
}

TOKEN_RESTORATION_MAP = {
    "__token_cpp__": "c++",
    "__token_csharp__": "c#",
    "__token_dotnet__": ".net",
    "__token_nodejs__": "node.js",
    "__token_vuejs__": "vue.js",
    "__token_reactjs__": "react.js",
    "__token_cicd__": "ci/cd",
    "__token_sklearn__": "scikit-learn",
}

SECTION_HEADERS = {
    "summary": [
        "summary",
        "professional summary",
        "career summary",
        "executive summary",
        "profile",
        "about me",
        "career objective",
        "objective",
        "personal summary",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "skills & competencies",
        "technologies",
        "technical expertise",
        "competencies",
        "core competencies",
        "tech stack",
        "tools & technologies",
        "technical stack",
        "areas of expertise",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "internship experience",
        "internships",
        "work history",
        "career history",
        "professional background",
        "relevant experience",
        "employment",
    ],
    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "educational qualifications",
        "education & credentials",
        "academics",
        "academic history",
    ],
    "projects": [
        "projects",
        "academic projects",
        "key projects",
        "personal projects",
        "technical projects",
        "selected projects",
        "portfolio",
        "notable projects",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses & certifications",
        "courses and certifications",
        "courses & certifications",
        "credentials",
        "certifications & licenses",
        "certifications & courses",
    ],
    "general": [
        "additional information",
        "additional details",
        "other details",
        "positions of responsibility",
        "responsibility",
        "personal details",
        "personal profile",
        "awards & achievements",
        "awards",
        "achievements",
        "extracurricular activities",
        "extra-curricular activities",
        "languages",
        "interests",
        "hobbies",
    ],
}


def clean_text(raw_text: str) -> str:
    """
    Cleans raw resume text:
    1. Converts text to lowercase.
    2. Protects special symbols such as C++, C#, .NET, and CI/CD.
    3. Strips URLs, emails, phone numbers for fair evaluation.
    4. Removes non-alphanumeric punctuation while retaining protected tokens.
    5. Normalizes whitespace and returns clean normalized text.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text.lower()

    # Mask email addresses and urls
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Protect key programming language terms
    for pattern, placeholder in TOKEN_PROTECTION_MAP.items():
        text = re.sub(pattern, placeholder, text)

    # Remove non-word characters except underscores, hyphens, and whitespace
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Restore protected symbols
    for placeholder, original in TOKEN_RESTORATION_MAP.items():
        text = text.replace(placeholder.lower(), original)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sections(raw_text: str) -> Dict[str, str]:
    """
    Splits resume text into standardized functional sections:
    'summary', 'skills', 'experience', 'education', 'projects', 'certifications', 'general'.
    """
    sections: Dict[str, str] = {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "certifications": "",
        "general": "",
    }

    if not raw_text:
        return sections

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    current_section = "general"
    section_buffers: Dict[str, List[str]] = {sec: [] for sec in sections}

    for line in lines:
        cleaned_line = re.sub(r"^[\s#*\-_=~]+", "", line.lower()).strip()
        header_candidate = re.split(r"[:\-|–—]", cleaned_line)[0].strip(" -#*=_~[](){}")

        # Protect inline attributes when inside projects or experience
        is_inline_proj_attribute = (
            current_section in ("projects", "experience")
            and (
                cleaned_line.startswith(("tech stack:", "key skills:", "tools:", "technologies:"))
                or line.startswith(("•", "-", "*"))
                or len(line) > 50
            )
        )

        found_header = None
        inline_content = ""

        if not is_inline_proj_attribute:
            # Check standalone or prefixed headers
            for sec_name, headers in SECTION_HEADERS.items():
                if header_candidate in headers and len(header_candidate) >= 3:
                    found_header = sec_name
                    # If line has content after colon/dash, extract it
                    parts = re.split(r"[:\-|–—]", line, maxsplit=1)
                    if len(parts) > 1 and parts[1].strip():
                        inline_content = parts[1].strip()
                    break
                elif any(
                    cleaned_line == h
                    or (cleaned_line.startswith(h + " ") and len(cleaned_line) < 40)
                    for h in headers
                ):
                    found_header = sec_name
                    break

        if found_header:
            current_section = found_header
            if inline_content:
                section_buffers[current_section].append(inline_content)
        else:
            section_buffers[current_section].append(line)

    for sec in sections:
        sections[sec] = "\n".join(section_buffers[sec])

    return sections

