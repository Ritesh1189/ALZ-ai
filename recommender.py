"""
Module: recommender.py
Computes job role matches using a hybrid scoring algorithm combining
TF-IDF semantic cosine similarity and hard skill set coverage.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


JOB_ROLES_PATH = Path(__file__).resolve().parent / "data" / "job_roles.csv"


class JobRecommender:
    """
    JobRecommender matches candidate resumes against curated job roles.
    Features:
    - TF-IDF unigram + bigram cosine similarity on comprehensive job descriptions
    - Exact technical skill recall & overlap scoring
    - Configurable hybrid score weighting
    """

    def __init__(self, roles_path: Optional[Path] = None):
        self.roles_path = roles_path or JOB_ROLES_PATH
        self.roles: List[Dict[str, Any]] = []
        self.skill_weight: float = 0.55
        self.semantic_weight: float = 0.45
        self._load_roles()

    def _load_roles(self) -> None:
        """
        Loads job role definitions from CSV and formats required skills lists.
        """
        if not self.roles_path.exists():
            raise FileNotFoundError(f"Job roles CSV not found at {self.roles_path}")

        self.roles = []
        with open(self.roles_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                req_skills_raw = row.get("required_skills", "").strip()
                skills_list = [
                    s.strip().lower() for s in req_skills_raw.split(",") if s.strip()
                ]

                role_entry = {
                    "role_id": int(row["role_id"]),
                    "role_name": row["role_name"].strip(),
                    "category": row["category"].strip(),
                    "required_skills": skills_list,
                    "required_skills_set": set(skills_list),
                    "description": row["description"].strip(),
                    # Comprehensive corpus for semantic TF-IDF matching
                    "combined_text": f"{row['role_name']} {row['category']} {' '.join(skills_list)} {row['description']}",
                }
                self.roles.append(role_entry)

    def get_all_roles(self) -> List[Dict[str, Any]]:
        """Returns all loaded job roles."""
        return self.roles

    def get_role_by_id(self, role_id: int) -> Optional[Dict[str, Any]]:
        """Look up a role by its numeric identifier."""
        for role in self.roles:
            if role["role_id"] == role_id:
                return role
        return None

    def get_role_by_name(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Look up a role by name (case-insensitive)."""
        lower = role_name.strip().lower()
        for role in self.roles:
            if role["role_name"].lower() == lower:
                return role
        return None

    def match_roles(
        self,
        candidate_skills: List[str],
        resume_text: str,
        top_n: int = 5,
        skill_weight: float = 0.55,
        semantic_weight: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Matches candidate resume against all curated job roles using hybrid scoring.
        
        Args:
            candidate_skills: List of skills extracted from candidate resume.
            resume_text: Cleaned or raw resume text.
            top_n: Number of top recommendations to return.
            skill_weight: Weight for exact skill overlap (0.0 to 1.0).
            semantic_weight: Weight for TF-IDF cosine similarity (0.0 to 1.0).
            
        Returns:
            List of ranked job match dictionaries.
        """
        if not self.roles:
            return []

        # Normalize weights so they sum to 1.0
        total_w = skill_weight + semantic_weight
        if total_w <= 0:
            skill_weight, semantic_weight = 0.5, 0.5
        else:
            skill_weight = skill_weight / total_w
            semantic_weight = semantic_weight / total_w

        candidate_skills_set: Set[str] = set(s.lower() for s in candidate_skills)

        # Prepare corpus for TF-IDF vectorization:
        # [0]: Candidate resume enriched with candidate skills
        # [1..N]: Job role combined descriptions
        enriched_candidate_text = f"{resume_text} {' '.join(candidate_skills_set)}"
        corpus = [enriched_candidate_text] + [r["combined_text"] for r in self.roles]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        candidate_vec = tfidf_matrix[0]
        role_vecs = tfidf_matrix[1:]

        # Compute cosine similarities between candidate and all roles
        cosine_sims = cosine_similarity(candidate_vec, role_vecs)[0]

        recommendations = []
        for idx, role in enumerate(self.roles):
            role_req_set = role["required_skills_set"]
            matched_skills = sorted(list(candidate_skills_set.intersection(role_req_set)))
            missing_skills = sorted(list(role_req_set.difference(candidate_skills_set)))

            # Skill overlap score (recall of required skills)
            total_req = len(role_req_set)
            skill_score = (len(matched_skills) / total_req) if total_req > 0 else 0.0

            # Semantic TF-IDF similarity score (clamped to [0.0, 1.0])
            semantic_score = float(np.clip(cosine_sims[idx], 0.0, 1.0))

            # Hybrid score
            hybrid_score = (skill_weight * skill_score) + (semantic_weight * semantic_score)
            match_percentage = round(hybrid_score * 100, 1)

            # Match level category
            if match_percentage >= 75.0:
                match_level = "High Match"
                badge_color = "#10b981"  # Emerald green
            elif match_percentage >= 50.0:
                match_level = "Moderate Match"
                badge_color = "#f59e0b"  # Amber
            else:
                match_level = "Low Match"
                badge_color = "#ef4444"  # Red

            recommendations.append({
                "role_id": role["role_id"],
                "role_name": role["role_name"],
                "category": role["category"],
                "match_score": match_percentage,
                "skill_score": round(skill_score * 100, 1),
                "semantic_score": round(semantic_score * 100, 1),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "required_skills": role["required_skills"],
                "matched_count": len(matched_skills),
                "required_count": total_req,
                "coverage_pct": round((len(matched_skills) / total_req * 100), 1) if total_req else 0,
                "match_level": match_level,
                "badge_color": badge_color,
                "description": role["description"],
            })

        # Sort descending by match_score
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)

        return recommendations[:top_n]

    def evaluate_role(
        self,
        role_identifier: Any,
        candidate_skills: List[str],
        resume_text: str,
        skill_weight: float = 0.55,
        semantic_weight: float = 0.45,
    ) -> Optional[Dict[str, Any]]:
        """
        Detailed evaluation for a specific selected role (by ID or name).
        """
        target_role = None
        if isinstance(role_identifier, int):
            target_role = self.get_role_by_id(role_identifier)
        elif isinstance(role_identifier, str):
            target_role = self.get_role_by_name(role_identifier)

        if not target_role:
            return None

        # Run match_roles for all and extract the specific role
        all_matches = self.match_roles(
            candidate_skills=candidate_skills,
            resume_text=resume_text,
            top_n=len(self.roles),
            skill_weight=skill_weight,
            semantic_weight=semantic_weight,
        )

        for match in all_matches:
            if match["role_id"] == target_role["role_id"]:
                return match

        return None

    def evaluate_custom_jd(
        self,
        job_title: str,
        job_description: str,
        candidate_skills: List[str],
        resume_text: str,
        extracted_jd_skills: Optional[List[str]] = None,
        skill_weight: float = 0.55,
        semantic_weight: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate's standardized resume against an ad-hoc custom job description.
        """
        total_w = skill_weight + semantic_weight
        if total_w <= 0:
            skill_weight, semantic_weight = 0.5, 0.5
        else:
            skill_weight = skill_weight / total_w
            semantic_weight = semantic_weight / total_w

        candidate_skills_set = set(s.lower() for s in candidate_skills)

        if extracted_jd_skills is None:
            from skill_extractor import get_skill_extractor
            jd_extractor = get_skill_extractor()
            jd_skills_res = jd_extractor.extract_skills(job_description)
            jd_req_skills = jd_skills_res.get("matched_skills", [])
        else:
            jd_req_skills = [s.lower() for s in extracted_jd_skills]

        jd_req_set = set(s.lower() for s in jd_req_skills)

        matched_skills = sorted(list(candidate_skills_set.intersection(jd_req_set)))
        missing_skills = sorted(list(jd_req_set.difference(candidate_skills_set)))

        total_req = len(jd_req_set)
        skill_score = (len(matched_skills) / total_req) if total_req > 0 else (1.0 if not jd_req_set else 0.0)

        enriched_candidate = f"{resume_text} {' '.join(candidate_skills_set)}"
        corpus = [enriched_candidate, job_description]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_mat = vectorizer.fit_transform(corpus)
        cosine_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        semantic_score = float(np.clip(cosine_sim, 0.0, 1.0))

        hybrid_score = (skill_weight * skill_score) + (semantic_weight * semantic_score)
        match_percentage = round(hybrid_score * 100, 1)

        if match_percentage >= 75.0:
            match_level = "High Match"
            badge_color = "#10b981"
        elif match_percentage >= 50.0:
            match_level = "Moderate Match"
            badge_color = "#f59e0b"
        else:
            match_level = "Low Match"
            badge_color = "#ef4444"

        return {
            "job_title": job_title or "Custom Job Specification",
            "match_score": match_percentage,
            "skill_score": round(skill_score * 100, 1),
            "semantic_score": round(semantic_score * 100, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "required_skills": sorted(list(jd_req_set)),
            "match_level": match_level,
            "badge_color": badge_color,
        }


# Module singleton
_default_recommender: Optional[JobRecommender] = None


def get_job_recommender() -> JobRecommender:
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = JobRecommender()
    return _default_recommender
