"""
Script: test_pipeline.py
Quick self-test script validating all core backend components:
- text_cleaner
- resume_parser
- skill_extractor
- recommender
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from text_cleaner import clean_text, extract_sections
from resume_parser import parse_resume, validate_file
from skill_extractor import get_skill_extractor
from recommender import get_job_recommender


SAMPLE_RESUME = """
John Doe - Senior Machine Learning & AI Engineer
john.doe@email.com | +1 555-0199 | San Francisco, CA

SUMMARY
Passionate AI Engineer with 4+ years of experience designing scalable machine learning pipelines,
fine-tuning Transformers, building RAG applications, and deploying PyTorch models using FastAPI and Docker in AWS.

SKILLS
Programming Languages: Python, C++, SQL, Bash
Machine Learning: PyTorch, Scikit-Learn, Deep Learning, Transformers, Hugging Face, LLM, RAG
DevOps & Cloud: Docker, Kubernetes, AWS, Git, CI/CD, Linux, MLOps, FastAPI

WORK EXPERIENCE
Senior Machine Learning Engineer | TechNova AI (2022 - Present)
- Architected and deployed an enterprise Retrieval-Augmented Generation (RAG) platform using PyTorch, LangChain, and vector databases, reducing document query latency by 45%.
- Fine-tuned transformer models (BERT, Llama) with Hugging Face for named entity recognition with 94% F1-score.
- Containerized microservices using Docker and automated CI/CD deployment pipelines on AWS Kubernetes cluster.

Machine Learning Engineer | DataCore Systems (2020 - 2022)
- Built predictive customer churn models with Scikit-Learn and XGBoost, saving $1.2M annually.
- Conducted exploratory data analysis (EDA), feature engineering, and model evaluation using Pandas and NumPy.

EDUCATION
Master of Science in Computer Science | Stanford University (2020)
Bachelor of Science in Computer Engineering | UC Berkeley (2018)

PROJECTS
OpenSource RAG Agent: Built an open-source multimodal conversational assistant with PyTorch, FastAPI, and Docker.
"""


def run_tests():
    print("--- 1. Testing text_cleaner ---")
    cleaned = clean_text("Experienced in C++, C#, .NET, and Python.")
    print("Cleaned text:", cleaned)
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert ".net" in cleaned
    assert "python" in cleaned
    print("-> text_cleaner passed!\n")

    print("--- 2. Testing resume_parser ---")
    is_valid, msg = validate_file("resume.txt", len(SAMPLE_RESUME.encode("utf-8")))
    assert is_valid, msg
    parsed = parse_resume("sample_john_doe.txt", SAMPLE_RESUME.encode("utf-8"))
    print(f"Format: {parsed['format']}, Words: {parsed['word_count']}")
    print(f"Detected sections: {list(parsed['sections'].keys())}")
    assert parsed["word_count"] > 100
    print("-> resume_parser passed!\n")

    print("--- 3. Testing skill_extractor ---")
    extractor = get_skill_extractor()
    skills_res = extractor.extract_skills(SAMPLE_RESUME)
    extracted_skills = skills_res["matched_skills"]
    print(f"Extracted {len(extracted_skills)} skills: {extracted_skills}")
    assert "python" in extracted_skills
    assert "pytorch" in extracted_skills
    assert "docker" in extracted_skills
    assert "c++" in extracted_skills
    print("-> skill_extractor passed!\n")

    print("--- 4. Testing recommender ---")
    recommender = get_job_recommender()
    recommendations = recommender.match_roles(
        candidate_skills=extracted_skills,
        resume_text=cleaned,
        top_n=3,
    )
    print(f"Top {len(recommendations)} recommended roles:")
    for i, r in enumerate(recommendations, 1):
        print(f"  {i}. {r['role_name']} - Match: {r['match_score']}% (Skill Score: {r['skill_score']}%, TFIDF: {r['semantic_score']}%)")
        print(f"     Matched ({len(r['matched_skills'])}): {r['matched_skills']}")
        print(f"     Missing ({len(r['missing_skills'])}): {r['missing_skills']}")

    top_role = recommendations[0]["role_name"]
    print(f"\nTop Match Role: {top_role}")
    assert top_role in ["Machine Learning Engineer", "AI Engineer"], f"Unexpected top role: {top_role}"
    print("-> recommender passed!\n")

    print("--- 5. Testing ats_scorer ---")
    from ats_scorer import get_ats_scorer
    ats_eval = get_ats_scorer()
    ats_res = ats_eval.evaluate_resume(
        raw_text=parsed["raw_text"],
        cleaned_text=parsed["cleaned_text"],
        sections=parsed["sections"],
        extracted_skills=extracted_skills,
        word_count=parsed["word_count"],
    )
    print(f"ATS Score: {ats_res['overall_score']}/100, Grade: {ats_res['grade']} ({ats_res['tier']})")
    assert ats_res["overall_score"] > 50, f"ATS score unexpectedly low: {ats_res['overall_score']}"
    assert "breakdown" in ats_res
    assert len(ats_res["suggestions"]) > 0
    print("-> ats_scorer passed!\n")

    print("--- 6. Testing gap_analyzer ---")
    from gap_analyzer import get_gap_analyzer
    gap_eval = get_gap_analyzer()
    target_role = recommender.get_role_by_id(recommendations[0]["role_id"])
    gap_res = gap_eval.analyze_gaps(target_role, extracted_skills)
    print(f"Gap Analysis for {gap_res['role_name']}: Coverage: {gap_res['coverage_pct']}%")
    print(f"Roadmap Phases: {len(gap_res['roadmap_phases'])}")
    assert gap_res["coverage_pct"] >= 50.0
    assert len(gap_res["roadmap_phases"]) > 0
    print("-> gap_analyzer passed!\n")

    print("ALL 6 CORE BACKEND MODULE TESTS PASSED PERFECTLY!")


if __name__ == "__main__":
    run_tests()
