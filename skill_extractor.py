"""
Module: skill_extractor.py
Extracts and categorizes technical and professional skills from resume text
using a curated skill dictionary with regular expression alias matching.
Properly handles special edge-case tokens including C++, C#, .NET, C, and R.
"""

import re
import csv
from pathlib import Path
from typing import Dict, List, Set, Any, Optional


SKILL_DICT_PATH = Path(__file__).resolve().parent / "data" / "skill_dictionary.csv"


class SkillExtractor:
    """
    SkillExtractor loads skill rules from CSV and performs high-precision
    keyword and alias matching against resume text and section blocks.
    """

    # Custom regex overrides for programming languages with special punctuation or single-letter names
    SPECIAL_TOKEN_PATTERNS = {
        "c++": re.compile(r"(?<![a-zA-Z0-9])(?:c\+\+|cpp)(?![a-zA-Z0-9+])", re.IGNORECASE),
        "c#": re.compile(r"(?<![a-zA-Z0-9])(?:c\#|csharp)(?![a-zA-Z0-9#])", re.IGNORECASE),
        "c": re.compile(r"(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])", re.IGNORECASE),
        "r": re.compile(r"(?<![a-zA-Z0-9])(?:r(?:\s+language|\s+programming)?)(?![a-zA-Z0-9&])", re.IGNORECASE),
        ".net": re.compile(r"(?<![a-zA-Z0-9])(?:\.net|dotnet|\.net\s+core|asp\.net)(?![a-zA-Z0-9])", re.IGNORECASE),
        "node.js": re.compile(r"(?<![a-zA-Z0-9])(?:node\.js|nodejs|node\s+js)(?![a-zA-Z0-9])", re.IGNORECASE),
        "vue": re.compile(r"(?<![a-zA-Z0-9])(?:vue(?:\.js|js)?)(?![a-zA-Z0-9])", re.IGNORECASE),
        "react": re.compile(r"(?<![a-zA-Z0-9])(?:react(?:\.js|js)?)(?![a-zA-Z0-9])", re.IGNORECASE),
        "ci/cd": re.compile(r"(?<![a-zA-Z0-9])(?:ci\/cd|cicd|continuous\s+integration)(?![a-zA-Z0-9])", re.IGNORECASE),
        "javascript": re.compile(r"(?<![.\w])(?:javascript|(?<!\.)\bjs\b)", re.IGNORECASE),
        "typescript": re.compile(r"(?<![.\w])(?:typescript|(?<!\.)\bts\b)", re.IGNORECASE),
        "go": re.compile(r"(?<![a-zA-Z0-9])(?:golang|go\s+lang(?:uage)?|\bgo\b)", re.IGNORECASE),
    }

    def __init__(self, dict_path: Optional[Path] = None):
        self.dict_path = dict_path or SKILL_DICT_PATH
        self.skills_data: List[Dict[str, Any]] = []
        self._load_dictionary()

    def _load_dictionary(self) -> None:
        """
        Loads the skills dictionary CSV and pre-compiles regex patterns.
        """
        if not self.dict_path.exists():
            raise FileNotFoundError(f"Skill dictionary not found at {self.dict_path}")

        self.skills_data = []
        with open(self.dict_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                skill_canonical = row["skill"].strip().lower()
                category = row["category"].strip()
                aliases_raw = row.get("aliases", "").strip()

                # Check if we have a hand-tuned special pattern
                if skill_canonical in self.SPECIAL_TOKEN_PATTERNS:
                    compiled_pattern = self.SPECIAL_TOKEN_PATTERNS[skill_canonical]
                    alias_list = [skill_canonical]
                    if aliases_raw:
                        alias_list.extend([a.strip().lower() for a in aliases_raw.split("|") if a.strip()])
                else:
                    alias_list = [skill_canonical]
                    if aliases_raw:
                        for alias in aliases_raw.split("|"):
                            alias = alias.strip().lower()
                            # Clean up unneeded literal escape characters from CSV if present
                            alias = alias.replace(r"\b", "").replace(r"\.", ".").replace(r"\+", "+").replace(r"\#", "#")
                            if alias and alias not in alias_list:
                                alias_list.append(alias)

                    # Escape all aliases safely and build standard word-boundary regex
                    patterns = [rf"\b{re.escape(alias)}\b" for alias in alias_list]
                    combined_pattern = re.compile(rf"({'|'.join(patterns)})", re.IGNORECASE)
                    compiled_pattern = combined_pattern

                self.skills_data.append({
                    "skill": skill_canonical,
                    "category": category,
                    "aliases": alias_list,
                    "compiled_pattern": compiled_pattern,
                })

    def extract_skills(self, text: str) -> Dict[str, Any]:
        """
        Scans input text and identifies all matching skills.
        Returns:
            - matched_skills: sorted list of canonical skill names
            - by_category: dict mapping category -> list of canonical skills
            - skill_frequencies: dict of canonical skill -> occurrence count
            - total_unique_skills: integer count of distinct skills
        """
        if not text or not isinstance(text, str):
            return {
                "matched_skills": [],
                "by_category": {},
                "categorized_skills": {},
                "skill_frequencies": {},
                "total_unique_skills": 0,
            }

        matched_skills_set: Set[str] = set()
        by_category: Dict[str, List[str]] = {}
        frequencies: Dict[str, int] = {}

        for entry in self.skills_data:
            canonical = entry["skill"]
            category = entry["category"]
            pattern: re.Pattern = entry["compiled_pattern"]

            matches = pattern.findall(text)
            if matches:
                count = len(matches)
                matched_skills_set.add(canonical)
                frequencies[canonical] = count
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(canonical)

        matched_sorted = sorted(list(matched_skills_set))
        for cat in by_category:
            by_category[cat] = sorted(by_category[cat])

        return {
            "matched_skills": matched_sorted,
            "by_category": by_category,
            "categorized_skills": by_category,
            "skill_frequencies": frequencies,
            "total_unique_skills": len(matched_sorted),
        }

    def extract_from_sections(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyzes each section independently to identify section-specific skill concentrations.
        """
        section_skill_map: Dict[str, List[str]] = {}
        all_skills: Set[str] = set()

        for sec_name, sec_text in sections.items():
            result = self.extract_skills(sec_text)
            section_skill_map[sec_name] = result["matched_skills"]
            all_skills.update(result["matched_skills"])

        return {
            "section_skills": section_skill_map,
            "all_extracted_skills": sorted(list(all_skills)),
        }


# Module singleton for performance
_default_extractor: Optional[SkillExtractor] = None


def get_skill_extractor() -> SkillExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = SkillExtractor()
    return _default_extractor
