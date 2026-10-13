from pathlib import Path
from resume_parser import parse_resume
from resume_standardizer import get_resume_standardizer
from skill_extractor import get_skill_extractor
from ats_scorer import get_ats_scorer
from recommender import get_job_recommender
from text_cleaner import clean_text, extract_sections

def run_tests():
    sample_path = Path("samples/ritesh_ture_resume.txt")
    with open(sample_path, "rb") as f:
        sample_bytes = f.read()

    parsed = parse_resume("ritesh_ture_resume.txt", sample_bytes)
    print("1. Parsed file successfully. Word count:", parsed["word_count"])

    std = get_resume_standardizer().standardize(parsed["raw_text"], parsed["file_name"], parsed["sections"])
    print("2. Standardized successfully! Name:", std["name"])
    print("   Formatted text length:", len(std["formatted_text"]))
    print("   HTML length:", len(std["html"]))

    skills_res = get_skill_extractor().extract_skills(std["formatted_text"])
    matched_skills = skills_res["matched_skills"]
    print("3. Extracted skills count from standardized resume:", len(matched_skills))
    print("   Skills:", matched_skills)

    cleaned_std = clean_text(std["formatted_text"])
    std_sections = extract_sections(std["formatted_text"])
    ats_res = get_ats_scorer().evaluate_resume(
        raw_text=std["formatted_text"],
        cleaned_text=cleaned_std,
        sections=std_sections,
        extracted_skills=matched_skills,
        word_count=len(std["formatted_text"].split()),
    )
    print(f"4. ATS Evaluation: Score = {ats_res['overall_score']}, Grade = {ats_res['grade']}")

    recommender = get_job_recommender()
    recs = recommender.match_roles(
        candidate_skills=matched_skills,
        resume_text=cleaned_std,
        top_n=3,
        skill_weight=0.6,
        semantic_weight=0.4,
    )
    print("5. Top Recommended Roles:")
    for r in recs:
        name = r["role_name"]
        m = r["match_score"]
        sk = r["skill_score"]
        sem = r["semantic_score"]
        print(f"   - {name}: {m}% match (Skill: {sk}%, Semantic: {sem}%)")

    # Test custom JD evaluation
    custom_jd = """
    We are looking for a Full Stack Software Engineer skilled in React.js, Spring Boot, Java, REST APIs, and MongoDB.
    Experience with Python, MySQL, and Docker is a plus.
    """
    custom_eval = recommender.evaluate_custom_jd(
        job_title="Senior Full Stack Engineer",
        job_description=custom_jd,
        candidate_skills=matched_skills,
        resume_text=cleaned_std,
    )
    print("6. Custom JD Evaluation:")
    print("   Title:", custom_eval["job_title"])
    print("   Match:", custom_eval["match_score"], "%")
    print("   Matched Skills:", custom_eval["matched_skills"])
    print("   Missing Skills:", custom_eval["missing_skills"])

    # Test user's actual uploaded PDF if present
    pdf_path = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\a3fdcc66-3df3-4659-8afa-857f510b2eff\.user_uploaded\media_1791910213941.pdf")
    if pdf_path.exists():
        print("\n--- 7. Testing User Uploaded PDF (LTM resume.pdf) ---")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        parsed_pdf = parse_resume("LTM resume.pdf", pdf_bytes)
        print("PDF Parsed successfully! Words:", parsed_pdf["word_count"])

        std_pdf = get_resume_standardizer().standardize(parsed_pdf["raw_text"], "LTM resume.pdf", parsed_pdf["sections"])
        print("Standardized PDF Profile Name:", std_pdf["name"])
        print("Standardized PDF Skills:", std_pdf["skills"])

        cleaned_pdf_text = clean_text(std_pdf["formatted_text"])
        pdf_recs = recommender.match_roles(
            candidate_skills=std_pdf["skills"],
            resume_text=cleaned_pdf_text,
            top_n=3,
        )
        print("PDF Top Matched Roles:")
        for r in pdf_recs:
            print("   -", r["role_name"], ":", r["match_score"], "%")

    # Test user's actual uploaded DOCX if present
    docx_path = Path(r"C:\Users\HP\Downloads\Aditya Kale Resume.docx")
    if docx_path.exists():
        print("\n--- 8. Testing User Uploaded DOCX (Aditya Kale Resume.docx) ---")
        with open(docx_path, "rb") as f:
            docx_bytes = f.read()
        parsed_docx = parse_resume("Aditya Kale Resume.docx", docx_bytes)
        print("DOCX Parsed successfully! Words:", parsed_docx["word_count"])

        std_docx = get_resume_standardizer().standardize(parsed_docx["raw_text"], "Aditya Kale Resume.docx", parsed_docx["sections"])
        print("Standardized DOCX Profile Name:", std_docx["name"])
        print("Role Subtitle:", std_docx["profile"]["role_subtitle"])
        print("Education entries count:", len(std_docx["profile"]["education"]))
        for edu in std_docx["profile"]["education"]:
            print("  -", edu["degree"], "|", edu["institution"], "|", edu["duration"], "|", edu["grade_label"])
        print("Projects count:", len(std_docx["profile"]["projects"]))
        for proj in std_docx["profile"]["projects"]:
            print("  -", proj["title"], "| Stack:", proj.get("tech_stack", ""), "| Bullets:", len(proj.get("bullets", [])))
        print("Certifications count:", len(std_docx["profile"]["certifications"]))
        for cert in std_docx["profile"]["certifications"]:
            print("  -", cert)
        print("Responsibility Title:", std_docx["profile"]["responsibility"].get("title", ""))

        cleaned_docx_text = clean_text(std_docx["formatted_text"])
        docx_sections = extract_sections(std_docx["formatted_text"])
        ats_docx = get_ats_scorer().evaluate_resume(
            raw_text=std_docx["formatted_text"],
            cleaned_text=cleaned_docx_text,
            sections=docx_sections,
            extracted_skills=std_docx["skills"],
            word_count=len(std_docx["formatted_text"].split()),
        )
        print(f"DOCX ATS Evaluation: Score = {ats_docx['overall_score']}, Grade = {ats_docx['grade']}")

    print("\nALL PIPELINE TESTS SUCCEEDED FLAWLESSLY!")

if __name__ == "__main__":
    run_tests()
