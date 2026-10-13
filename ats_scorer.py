"""
Module: ats_scorer.py
Evaluates resumes for Applicant Tracking System (ATS) compatibility, readability,
action verb density, quantifiable metrics, section structure, and technical keyword density.
"""

import re
from typing import Dict, List, Any


STRONG_ACTION_VERBS = {
    "architected", "accelerated", "accomplished", "achieved", "administered",
    "advanced", "analyzed", "automated", "built", "centralized", "championed",
    "collaborated", "compiled", "composed", "configured", "constructed",
    "coordinated", "created", "customized", "debugged", "decreased", "delivered",
    "deployed", "designed", "developed", "devised", "diagnosed", "directed",
    "documented", "doubled", "drafted", "drove", "eliminated", "enabled",
    "engineered", "enhanced", "established", "evaluated", "exceeded", "executed",
    "expanded", "expedited", "fabricated", "facilitated", "fine-tuned", "formulated",
    "founded", "generated", "guided", "implemented", "improved", "increased",
    "initiated", "innovated", "inspected", "installed", "integrated", "invented",
    "investigated", "launched", "led", "leveraged", "managed", "maximized",
    "mentored", "minimized", "modernized", "negotiated", "optimized", "orchestrated",
    "overhauled", "oversaw", "partnered", "pioneered", "planned", "produced",
    "programmed", "promoted", "published", "re-engineered", "rebuilt", "reduced",
    "refactored", "resolved", "restructured", "revamped", "saved", "scaled",
    "scheduled", "secured", "simplified", "spearheaded", "standardized", "streamlined",
    "strengthened", "supervised", "surpassed", "synthesized", "trained", "transformed",
    "tripled", "troubleshot", "upgraded", "validated", "verified"
}

WEAK_PHRASES = [
    "responsible for", "duties included", "helped with", "worked on",
    "assisted in", "participated in", "tasked with", "handled"
]


class ATSScorer:
    """
    Evaluates resumes against industry-standard ATS filtering algorithms.
    """

    def evaluate_resume(
        self,
        raw_text: str,
        cleaned_text: str,
        sections: Dict[str, str],
        extracted_skills: List[str],
        word_count: int,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive ATS evaluation and returns itemized scores, grades, and tips.
        """
        text_lower = raw_text.lower()
        words = cleaned_text.split()
        total_words = len(words) if words else word_count

        # 1. Section Structure Score (Max 25 pts)
        section_score, section_details = self._evaluate_sections(sections)

        # 2. Action Verbs Score (Max 20 pts)
        action_verb_score, action_verb_details = self._evaluate_action_verbs(words)

        # 3. Measurable Impact & Quantifiable Results (Max 20 pts)
        metric_score, metric_details = self._evaluate_quantifiables(raw_text)

        # 4. Word Count & Length Health (Max 15 pts)
        length_score, length_details = self._evaluate_length(total_words)

        # 5. Contact Information & Professional Links (Max 10 pts)
        contact_score, contact_details = self._evaluate_contact_info(raw_text)

        # 6. Skill Keyword Density (Max 10 pts)
        skill_density_score, skill_density_details = self._evaluate_skill_density(
            len(extracted_skills), total_words
        )

        # Total ATS Score (0 - 100)
        total_ats_score = round(
            section_score
            + action_verb_score
            + metric_score
            + length_score
            + contact_score
            + skill_density_score
        )
        total_ats_score = max(0, min(100, total_ats_score))

        # Check for weak passive phrases
        found_weak_phrases = []
        for phrase in WEAK_PHRASES:
            if phrase in text_lower:
                found_weak_phrases.append(phrase)

        # Grade & Assessment
        if total_ats_score >= 85:
            grade = "A"
            tier = "Excellent (ATS Ready)"
            color = "#10b981"
        elif total_ats_score >= 70:
            grade = "B"
            tier = "Strong Candidate"
            color = "#3b82f6"
        elif total_ats_score >= 55:
            grade = "C"
            tier = "Moderate (Needs Optimization)"
            color = "#f59e0b"
        else:
            grade = "D"
            tier = "Low (High Rejection Risk)"
            color = "#ef4444"

        # Construct specific suggestions for improvement
        suggestions = self._build_suggestions(
            section_details=section_details,
            action_verb_details=action_verb_details,
            metric_details=metric_details,
            length_details=length_details,
            contact_details=contact_details,
            skill_density_details=skill_density_details,
            weak_phrases=found_weak_phrases,
        )

        return {
            "overall_score": total_ats_score,
            "grade": grade,
            "tier": tier,
            "color": color,
            "breakdown": {
                "section_structure": {
                    "score": round(section_score),
                    "max": 25,
                    "details": section_details,
                },
                "action_verbs": {
                    "score": round(action_verb_score),
                    "max": 20,
                    "details": action_verb_details,
                },
                "quantifiable_impact": {
                    "score": round(metric_score),
                    "max": 20,
                    "details": metric_details,
                },
                "length_hygiene": {
                    "score": round(length_score),
                    "max": 15,
                    "details": length_details,
                },
                "length_health": {
                    "score": round(length_score),
                    "max": 15,
                    "details": length_details,
                },
                "contact_info": {
                    "score": round(contact_score),
                    "max": 10,
                    "details": contact_details,
                },
                "skill_density": {
                    "score": round(skill_density_score),
                    "max": 10,
                    "details": skill_density_details,
                },
            },
            "weak_phrases_detected": found_weak_phrases,
            "suggestions": suggestions,
        }

    def _evaluate_sections(self, sections: Dict[str, str]) -> tuple[float, Dict[str, Any]]:
        score = 0.0
        details = {}
        # Skills section (8 pts)
        has_skills = bool(sections.get("skills", "").strip())
        score += 8.0 if has_skills else 0.0
        details["skills_section"] = has_skills

        # Experience section (8 pts)
        has_exp = bool(sections.get("experience", "").strip())
        score += 8.0 if has_exp else 0.0
        details["experience_section"] = has_exp

        # Education section (5 pts)
        has_edu = bool(sections.get("education", "").strip())
        score += 5.0 if has_edu else 0.0
        details["education_section"] = has_edu

        # Projects section (2 pts)
        has_proj = bool(sections.get("projects", "").strip())
        score += 2.0 if has_proj else 0.0
        details["projects_section"] = has_proj

        # Certifications section (2 pts)
        has_cert = bool(sections.get("certifications", "").strip())
        score += 2.0 if has_cert else 0.0
        details["certifications_section"] = has_cert

        return score, details

    def _evaluate_action_verbs(self, words: List[str]) -> tuple[float, Dict[str, Any]]:
        detected_verbs = set()
        for w in words:
            w_clean = re.sub(r"[^\w]", "", w.lower())
            if w_clean in STRONG_ACTION_VERBS:
                detected_verbs.add(w_clean)

        count = len(detected_verbs)
        # Scaled score: 10+ distinct verbs = 20 pts
        if count >= 10:
            score = 20.0
        elif count >= 7:
            score = 16.0
        elif count >= 4:
            score = 12.0
        elif count >= 2:
            score = 8.0
        elif count == 1:
            score = 4.0
        else:
            score = 0.0

        details = {
            "action_verb_count": count,
            "action_verbs_count": count,
            "found_verbs": sorted(list(detected_verbs)),
            "action_verbs_sample": sorted(list(detected_verbs))[:12],
        }
        return score, details

    def _evaluate_quantifiables(self, raw_text: str) -> tuple[float, Dict[str, Any]]:
        # Check percentage mentions
        pct_matches = re.findall(r"\b\d+(?:\.\d+)?%", raw_text)
        # Check currency mentions ($100k, $1.5M, etc)
        money_matches = re.findall(r"[\$€£]\s*\d+(?:\.\d+)?[kKmMbB]?", raw_text)
        # Check multiplier / scale metrics (e.g. 10x, 2x, 50k users, 99.9%)
        scale_matches = re.findall(r"\b\d+(?:\.\d+)?(?:x|k|\+)\b", raw_text, re.IGNORECASE)

        total_metrics = len(pct_matches) + len(money_matches) + len(scale_matches)

        if total_metrics >= 5:
            score = 20.0
        elif total_metrics >= 3:
            score = 16.0
        elif total_metrics >= 2:
            score = 12.0
        elif total_metrics == 1:
            score = 7.0
        else:
            score = 2.0

        details = {
            "metrics_count": total_metrics,
            "metrics_found": pct_matches + money_matches + scale_matches,
            "percentages_found": pct_matches[:5],
            "financial_metrics_found": money_matches[:5],
        }
        return score, details

    def _evaluate_length(self, total_words: int) -> tuple[float, Dict[str, Any]]:
        # Optimal 1-2 page resume length: 350 - 900 words
        if 350 <= total_words <= 900:
            score = 15.0
            status = "Optimal Length (1 - 2 pages)"
        elif 250 <= total_words < 350:
            score = 11.0
            status = "Slightly Short"
        elif 900 < total_words <= 1200:
            score = 11.0
            status = "Slightly Long"
        elif total_words < 250:
            score = 5.0
            status = "Too Short (Sparse Content)"
        else:
            score = 6.0
            status = "Too Long (Risk of ATS Truncation)"

        details = {
            "word_count": total_words,
            "length_status": status,
            "status": status,
        }
        return score, details

    def _evaluate_contact_info(self, raw_text: str) -> tuple[float, Dict[str, Any]]:
        has_email = bool(re.search(r"\S+@\S+\.\S+", raw_text))
        has_phone = bool(re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text))
        has_linkedin_or_github = bool(re.search(r"linkedin\.com|github\.com", raw_text, re.IGNORECASE))

        score = 0.0
        if has_email:
            score += 4.0
        if has_phone:
            score += 3.0
        if has_linkedin_or_github:
            score += 3.0

        details = {
            "has_email": has_email,
            "has_phone": has_phone,
            "has_professional_link": has_linkedin_or_github,
        }
        return score, details

    def _evaluate_skill_density(
        self, unique_skills_count: int, total_words: int
    ) -> tuple[float, Dict[str, Any]]:
        if total_words == 0:
            return 0.0, {"density_pct": 0, "status": "No text"}

        density_pct = (unique_skills_count / total_words) * 100

        # Ideal technical density: 2.0% - 7.5%
        if 2.0 <= density_pct <= 7.5:
            score = 10.0
            status = "Balanced keyword density"
        elif density_pct > 7.5:
            score = 6.0
            status = "Potential keyword stuffing"
        elif 1.0 <= density_pct < 2.0:
            score = 7.0
            status = "Slightly low skill density"
        else:
            score = 3.0
            status = "Insufficient technical skills mentioned"

        details = {
            "unique_skills": unique_skills_count,
            "density_pct": round(density_pct, 1),
            "status": status,
        }
        return score, details

    def _build_suggestions(
        self,
        section_details: Dict[str, Any],
        action_verb_details: Dict[str, Any],
        metric_details: Dict[str, Any],
        length_details: Dict[str, Any],
        contact_details: Dict[str, Any],
        skill_density_details: Dict[str, Any],
        weak_phrases: List[str],
    ) -> List[str]:
        suggestions = []

        # Sections
        if not section_details.get("skills_section"):
            suggestions.append("Add a clearly labeled 'Skills' or 'Technical Skills' section heading.")
        if not section_details.get("experience_section"):
            suggestions.append("Add a dedicated 'Work Experience' or 'Professional Experience' section.")
        if not section_details.get("education_section"):
            suggestions.append("Include an 'Education' or 'Academic Background' section.")
        if not section_details.get("projects_section"):
            suggestions.append("Include a 'Projects' section highlighting real-world technical implementations.")

        # Action verbs
        verb_cnt = action_verb_details.get("action_verb_count", 0)
        if verb_cnt < 6:
            suggestions.append(
                f"Incorporate more strong action verbs (found {verb_cnt}). Use words like 'Architected', 'Spearheaded', 'Optimized', and 'Deployed' at the start of bullet points."
            )

        # Quantifiable metrics
        metric_cnt = metric_details.get("metrics_count", 0)
        if metric_cnt < 3:
            suggestions.append(
                "Quantify your achievements with numbers, percentages, or scale metrics (e.g., 'reduced latency by 35%', 'managed $500K budget', 'served 100K+ daily active users')."
            )

        # Weak phrases
        if weak_phrases:
            suggestions.append(
                f"Replace passive phrases ({', '.join(weak_phrases[:3])}) with direct, high-impact action verbs."
            )

        # Length
        word_cnt = length_details.get("word_count", 0)
        if word_cnt < 300:
            suggestions.append(
                f"Your resume is relatively short ({word_cnt} words). Expand on technical challenges solved, tools used, and measurable results."
            )
        elif word_cnt > 1100:
            suggestions.append(
                f"Your resume is quite lengthy ({word_cnt} words). Consider condensing bullet points to fit the standard 1 to 2 page format."
            )

        # Contact info
        if not contact_details.get("has_email"):
            suggestions.append("Ensure your primary professional email address is prominently visible at the top.")
        if not contact_details.get("has_linkedin_or_github"):
            suggestions.append("Add clickable links to your LinkedIn profile and GitHub repository.")

        # Keyword density
        if skill_density_details.get("density_pct", 0) < 1.5:
            suggestions.append("Incorporate more industry-relevant technical keywords and framework names.")

        if not suggestions:
            suggestions.append("Excellent ATS layout! Your resume meets key automated screening criteria.")

        return suggestions


# Singleton instance
_default_ats_scorer: ATSScorer = ATSScorer()


def get_ats_scorer() -> ATSScorer:
    global _default_ats_scorer
    return _default_ats_scorer
