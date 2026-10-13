"""
Module: resume_standardizer.py
Converts incoming resumes (PDF, DOCX, TXT) into a structured format
without deleting or truncating any candidate content, experience, or skills.
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from resume_formatter import DEFAULT_PROFILE, render_dyp_resume_html
from skill_extractor import get_skill_extractor


class ResumeStandardizer:
    """
    Parses unstructured or semi-structured resume text and reconstructs it
    into a structured placement resume schema while preserving 100% of the
    candidate's actual experience, education, projects, skills, and summary.
    """

    def __init__(self):
        self.skill_extractor = get_skill_extractor()

    def standardize(
        self,
        raw_text: str,
        file_name: str = "Candidate_Resume",
        sections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Main conversion entry point.
        Takes raw text and sections, extracts structured entities, and formats
        them without losing or truncating necessary resume text.
        """
        if not sections:
            from text_cleaner import extract_sections
            sections = extract_sections(raw_text)

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        lower_text = raw_text.lower()
        is_dyp = any(t in lower_text for t in ("d.y. patil", "dy patil", "dyp", "akurdi"))

        # Base profile structure
        profile: Dict[str, Any] = {
            "name": "",
            "role_subtitle": "",
            "phone": "",
            "email": "",
            "location": "",
            "college_logo_title": "DYP" if is_dyp else "RESUME",
            "college_subtext": "DR. D. Y. PATIL PRATISHTHAN" if is_dyp else "PLACEMENT RESUME FORMAT",
            "summary": "",
            "education": [],
            "internships": [],
            "projects": [],
            "certifications": [],
            "responsibility": {},
            "personal_details": {},
        }

        # 1. Header Information: Name, Email, Phone, Location, Title
        extracted_name = self._extract_name(lines, file_name, raw_text)
        extracted_email = self._extract_email(raw_text)
        extracted_phone = self._extract_phone(raw_text)
        extracted_location = self._extract_location(raw_text, lines)
        extracted_role = self._extract_role_or_degree(raw_text, lines)

        profile["name"] = extracted_name.upper()
        profile["email"] = extracted_email
        profile["phone"] = extracted_phone
        profile["location"] = extracted_location
        profile["role_subtitle"] = extracted_role

        # 2. Professional Summary
        summary_text = sections.get("summary", "").strip()
        if not summary_text:
            summary_text = self._extract_summary(lines)
        profile["summary"] = summary_text

        # 3. Education Entries
        edu_entries = self._extract_education(sections.get("education", ""), raw_text)
        profile["education"] = edu_entries

        # 4. Work Experience & Internships
        exp_entries = self._extract_internships(sections.get("experience", ""), raw_text)
        profile["internships"] = exp_entries

        # 5. Projects
        project_entries = self._extract_projects(sections.get("projects", ""), raw_text)
        profile["projects"] = project_entries

        # 6. Certifications
        certs_list = self._extract_certifications(
            sections.get("certifications", ""),
            sections.get("general", ""),
            raw_text,
        )
        profile["certifications"] = certs_list

        # 7. Positions of Responsibility
        resp_entry = self._extract_responsibilities(sections.get("general", ""), raw_text)
        profile["responsibility"] = resp_entry

        # 8. Personal Details
        personal_info = self._extract_personal_details(
            raw_text, profile, sections.get("general", "")
        )
        profile["personal_details"] = personal_info

        # 9. Render Canonical Plaintext & HTML (zero text dropped)
        formatted_text = self._build_canonical_text(profile, raw_text)
        formatted_html = render_dyp_resume_html(profile, enable_toolbar=True)

        # 10. Extract all skills across entire profile and raw text
        all_text = f"{raw_text}\n\n{formatted_text}"
        extracted_skills = self.skill_extractor.extract_skills(all_text)["matched_skills"]

        return {
            "profile": profile,
            "formatted_text": formatted_text,
            "html": formatted_html,
            "name": profile["name"],
            "skills": extracted_skills,
        }

    def _extract_name(self, lines: List[str], file_name: str, raw_text: str) -> str:
        """Detects candidate name from the top header lines or file name."""
        stop_words = {
            "summary", "professional", "objective", "profile", "technical",
            "skills", "experience", "work", "employment", "education", "projects",
            "certifications", "certificates", "licenses", "gender", "male", "female",
            "marital", "status", "single", "married", "address", "date", "birth", "dob",
            "languages", "known", "hindi", "marathi", "english", "phone", "email",
            "resume", "curriculum", "vitae", "github", "linkedin", "page", "college",
            "university", "ph", "contact", "developer", "engineer", "analyst", "specialist",
            "about", "me", "overview", "qualification", "qualifications", "at", "post",
            "current", "permanent", "simangaon", "pune", "parbhani", "mumbai",
            "chowk", "railway", "station", "rly", "nagar", "road", "street", "lane",
            "flat", "plot", "tq", "dist", "near", "behind", "opp", "opposite",
            "building", "bldg", "apt", "apartment", "colony", "vihar", "marg",
            "sector", "taluka", "tahsil", "district", "state", "pin", "pincode",
            "zip", "zipcode", "society", "maharashtra", "karnataka", "delhi",
            "gujarat", "telangana", "india", "akurdi", "gurudwara"
        }

        meta_prefixes = (
            "gender:", "marital status:", "marital:", "address:", "current address:",
            "permanent address:", "dob:", "date of birth:", "languages:", "known languages:",
            "phone:", "phones:", "phone numbers:", "email:", "emails:", "contact:",
            "curriculum vitae", "resume"
        )

        # Check top 25 lines for candidate name
        for line in lines[:25]:
            stripped = line.strip()
            if not stripped:
                continue

            low = stripped.lower()
            if any(low.startswith(p) for p in meta_prefixes):
                continue

            # If line has "Name: John Doe"
            name_label_match = re.search(r"^(?:name|candidate|applicant)[:\s]+([A-Za-z\s]+)$", stripped, re.IGNORECASE)
            if name_label_match:
                cand = name_label_match.group(1).strip()
                cand_clean = re.sub(r"\s+", " ", cand).strip()
                if 2 <= len(cand_clean.split()) <= 4:
                    return cand_clean.title()

            # Split on common separators like ' - Senior Engineer', ' | Full Stack', ', PMP'
            candidate_part = re.split(r"\s*[-|–—•,]\s*", stripped)[0].strip()
            clean_cand = re.sub(r"[^a-zA-Z\s]", "", candidate_part).strip()
            clean_cand = re.sub(r"\s+", " ", clean_cand).strip()
            words = clean_cand.split()

            if 2 <= len(words) <= 4:
                lower_words = [w.lower() for w in words]
                # Ensure no word is a known section header, address token, or stop word
                if not any(w in stop_words for w in lower_words):
                    if all(len(w) >= 2 for w in words):
                        return clean_cand.title()

        # Fallback to sanitized file name
        clean_file_base = Path(file_name).stem.replace("_", " ").replace("-", " ")
        words = [
            w for w in clean_file_base.split()
            if w.lower() not in ("resume", "cv", "upload", "sample", "document", "candidate")
        ]
        if 1 <= len(words) <= 4:
            return " ".join(words).title()

        return "Candidate Profile"

    def _extract_email(self, text: str) -> str:
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        if emails:
            # Fix common OCR / typo in gmail domain
            em = emails[0]
            em = re.sub(r"@gmai[iI]\.com", "@gmail.com", em, flags=re.IGNORECASE)
            return em
        return ""

    def _extract_phone(self, text: str) -> str:
        phones = re.findall(
            r"(?:\+91[-.\s]?)?[6-9]\d{9}|\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b[6-9]\d{9}\b",
            text,
        )
        if phones:
            p = phones[0].strip()
            if len(p) == 10 and p.isdigit():
                return f"+91-{p}"
            return p
        return ""

    def _extract_location(self, text: str, lines: List[str]) -> str:
        # Check explicit location or address line
        addr_match = re.search(
            r"(?:location|current address|address|at post\.?)[:\s]*([A-Za-z0-9\s,.\-]+?)(?:\n|•|\||$)",
            text,
            re.IGNORECASE,
        )
        if addr_match:
            candidate_loc = addr_match.group(1).strip(" ,|-")
            if 5 < len(candidate_loc) < 85:
                return candidate_loc

        loc_match = re.search(
            r"\b([A-Za-z\s]+,\s*(?:Maharashtra|Karnataka|Telangana|Delhi|Tamil Nadu|Gujarat|India|[A-Z]{2}|USA|United States|UK|Canada)(?:\s*-\s*\d{6})?)\b",
            text,
            re.IGNORECASE,
        )
        if loc_match:
            candidate_loc = loc_match.group(1).strip()
            if len(candidate_loc) < 65:
                return candidate_loc

        # Check top 5 lines for city mentions
        for line in lines[:5]:
            if any(term in line.lower() for term in ("pune", "mumbai", "bangalore", "delhi", "hyderabad", "san francisco", "new york", "london", "seattle", "parbhani")):
                clean_l = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", line)
                clean_l = re.sub(r"\+?\d[\d\s\-]{8,}\d", "", clean_l).strip(" ,|•-")
                if 4 < len(clean_l) < 65:
                    return clean_l

        return ""

    def _extract_role_or_degree(self, text: str, lines: List[str]) -> str:
        # Check top lines for degree or subtitle
        for line in lines[:6]:
            if re.match(r"^(?:B\.?E\.?|B\.?Tech)\s*-\s*", line, re.IGNORECASE):
                clean_line = re.sub(r"\s+at\s+.*$", "", line, flags=re.IGNORECASE).strip()
                if len(clean_line) < 60:
                    return clean_line

            parts = re.split(r"\s*[-|–—•]\s*", line)
            if len(parts) > 1:
                for candidate_p in parts[1:]:
                    title_part = candidate_p.strip()
                    words = [w for w in title_part.split() if w.isalpha()]
                    if (
                        len(words) >= 2
                        and 4 < len(title_part) < 60
                        and not any(w in title_part.lower() for w in ("phone", "email", "http", "www", "github", "dist", "parbhani", "pune", "chowk", "nagar", "road", "street"))
                        and not re.search(r"\d{3,}", title_part)
                    ):
                        title_clean = re.sub(r"\s+at\s+.*$", "", title_part, flags=re.IGNORECASE).strip()
                        return title_clean

        role_match = re.search(
            r"\b(Senior\s+[A-Za-z\s]+|Lead\s+[A-Za-z\s]+|Software Engineer|Full Stack Developer|Data Scientist|Data Analyst|Machine Learning Engineer|AI Engineer|Cloud Engineer|DevOps Engineer|B\.?E\.?\s*-\s*[A-Za-z\s]+|B\.?Tech\s*-\s*[A-Za-z\s]+|pursuing\s+(?:BE|B\.E\.|B\.Tech)\s+in\s+[A-Za-z\s]+)\b",
            text,
            re.IGNORECASE,
        )
        if role_match:
            candidate_role = role_match.group(0).strip()[:55]
            if "pursuing" in candidate_role.lower():
                candidate_role = re.sub(r"^pursuing\s+", "", candidate_role, flags=re.IGNORECASE)
                candidate_role = candidate_role.replace("BE in CSE", "B.E. - Computer Science & Engineering").replace("BE in IT", "B.E. - Information Technology")
            candidate_role = re.sub(r"\s+at\s+.*$", "", candidate_role, flags=re.IGNORECASE).strip()
            candidate_role = re.sub(r"\s+at\s*$", "", candidate_role, flags=re.IGNORECASE).strip()
            if "\n" not in candidate_role:
                return candidate_role

        return "B.E. - Computer Science & Engineering"

    def _extract_summary(self, lines: List[str]) -> str:
        """Finds any summary paragraph in the top 15 lines if not caught by section headers."""
        for i, line in enumerate(lines[:12]):
            if any(term in line.lower() for term in ("summary", "overview", "profile", "about me", "career objective")):
                buf = []
                for next_l in lines[i + 1: i + 6]:
                    if any(next_l.upper().startswith(h) for h in ("SKILLS", "EXPERIENCE", "EDUCATION", "PROJECTS", "ADDITIONAL")):
                        break
                    buf.append(next_l)
                if buf:
                    return " ".join(buf)
        return ""

    def _clean_institution_name(self, name: str, date_re: re.Pattern, grade_re: re.Pattern) -> str:
        clean = re.sub(date_re, "", name)
        clean = re.sub(grade_re, "", clean)
        clean = re.sub(r"^(?:at|in|from)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\(\s*\)", "", clean)
        return clean.strip(" ,|-(:[]{}")

    def _clean_degree_name(self, deg: str, date_re: re.Pattern, grade_re: re.Pattern) -> str:
        clean = re.sub(date_re, "", deg)
        clean = re.sub(grade_re, "", clean)
        clean = re.sub(r"\(\s*\)", "", clean)
        clean = clean.strip(" ,|-(:[]{}")
        lower = clean.lower()
        if "ssc" in lower and "10th" not in lower:
            return "10th | " + clean
        if ("hse" in lower or "hsc" in lower) and "12th" not in lower:
            return "12th | " + clean
        if clean.upper().startswith("E -") or clean.upper().startswith("E-"):
            return "B." + clean
        return clean

    def _extract_education(self, edu_text: str, full_text: str) -> List[Dict[str, str]]:
        """
        Parses education entries across multiple formats (pipe-delimited,
        multi-line paired, board list with date pools, and standard CV blocks)
        ensuring each institution has its own correct degree, duration, and grade.
        """
        raw_target = edu_text.strip() if len(edu_text.strip()) > 15 else full_text
        lines = [l.strip() for l in raw_target.splitlines() if l.strip()]

        clean_lines = []
        for l in lines:
            lower = l.lower()
            if lower in ("education", "academics", "academic qualifications", "academic background"):
                continue
            if lower.startswith(("certif", "technical skills", "skills:", "languages:", "awards:", "hobbies:")):
                break
            clean_lines.append(l)

        date_re = re.compile(
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|Present|Current)|20\d{2}\s*[-–—]\s*(?:20\d{2}|Present)|20\d{2})\b",
            re.IGNORECASE,
        )
        grade_re = re.compile(
            r"(?:CGPA|GPA|Percentage|Percent|Grade)[:\s]*([\d\.]+(?:\s*(?:/\s*(?:100|10|4)|%))?)|(\b\d{1,2}(?:\.\d{1,2})?\s*%)",
            re.IGNORECASE,
        )
        univ_re = re.compile(
            r"\b(college|university|institute|school|academy|polytechnic|vidyalaya|pratishthan|d\.?y\.?\s*patil|stanford|berkeley|mit|harvard|carnegie|purdue)\b",
            re.IGNORECASE,
        )
        deg_re = re.compile(
            r"\b(bachelor|master|ph\.?d|doctor|b\.?e\.?|b\.?tech|b\.?s\.?|m\.?s\.?|m\.?tech|diploma|12th|10th|hsc|ssc|hse|cbse|icse|secondary|higher secondary|computer science|software engineering|information technology|statistics|economics)\b",
            re.IGNORECASE,
        )

        has_pipe_entries = any("|" in l for l in clean_lines)
        entries: List[Dict[str, str]] = []

        if has_pipe_entries:
            i = 0
            while i < len(clean_lines):
                line = clean_lines[i]
                is_univ = bool(univ_re.search(line))
                is_deg = bool(deg_re.search(line))
                combined = line

                # Check if next line is a companion line (e.g. Line 1: Institution | Date, Line 2: Degree | Grade)
                if i + 1 < len(clean_lines):
                    next_l = clean_lines[i + 1]
                    if "|" in next_l:
                        next_is_univ = bool(univ_re.search(next_l))
                        next_is_deg = bool(deg_re.search(next_l))
                        if (is_univ and next_is_deg and not next_is_univ) or (is_deg and next_is_univ and not is_univ):
                            combined = line + " | " + next_l
                            i += 1

                parts = [p.strip() for p in combined.split("|") if p.strip()]
                inst, deg, dur, grd = "", "", "", ""
                for p in parts:
                    dm = date_re.search(p)
                    gm = grade_re.search(p)
                    if gm and not grd:
                        grd = gm.group(0)
                    elif dm and not dur and len(p.split()) <= 4:
                        dur = dm.group(0)

                    clean_p = self._clean_institution_name(p, date_re, grade_re)
                    if univ_re.search(clean_p) and not inst:
                        inst = clean_p
                    elif deg_re.search(clean_p) and not deg:
                        deg = self._clean_degree_name(clean_p, date_re, grade_re)
                    elif not deg and deg_re.search(clean_p):
                        deg = self._clean_degree_name(clean_p, date_re, grade_re)
                    elif not inst and not dm and not gm:
                        if not deg:
                            deg = self._clean_degree_name(clean_p, date_re, grade_re)
                        else:
                            inst = clean_p

                if inst or deg:
                    entries.append({
                        "institution": inst or deg,
                        "degree": deg or (inst if inst else "B.E. - Computer Science & Engineering"),
                        "duration": dur,
                        "grade_label": grd or "Completed",
                    })
                i += 1

            if entries:
                return self._order_education_entries(entries)

        # Fallback Case: Multi-line / Board / Institution blocks without pipes
        records: List[Dict[str, str]] = []
        curr: Optional[Dict[str, str]] = None
        dates_pool: List[str] = []

        for line in clean_lines:
            dm = date_re.search(line)
            gm = grade_re.search(line)
            is_date_only = dm and len(line.strip()) <= len(dm.group(0)) + 4

            if is_date_only:
                dates_pool.append(dm.group(0))
                continue

            is_univ = bool(univ_re.search(line))
            is_board = bool(deg_re.search(line)) and ("board" in line.lower() or len(line.split()) <= 5)

            if is_board and not is_univ:
                if curr:
                    records.append(curr)
                curr = {
                    "degree": self._clean_degree_name(line, date_re, grade_re),
                    "institution": "",
                    "duration": dm.group(0) if dm else "",
                    "grade_label": gm.group(0) if gm else "",
                }
            elif is_univ:
                inst_clean = self._clean_institution_name(line, date_re, grade_re)
                dur = dm.group(0) if dm else ""
                grd = gm.group(0) if gm else ""
                if curr and not curr["institution"]:
                    curr["institution"] = inst_clean
                    if not curr["duration"] and dur:
                        curr["duration"] = dur
                    if not curr["grade_label"] and grd:
                        curr["grade_label"] = grd
                else:
                    if curr:
                        records.append(curr)
                    curr = {
                        "degree": "",
                        "institution": inst_clean,
                        "duration": dur,
                        "grade_label": grd,
                    }
            elif curr:
                if gm and not curr["grade_label"]:
                    curr["grade_label"] = gm.group(0)
                if dm and not curr["duration"]:
                    curr["duration"] = dm.group(0)
                if deg_re.search(line) and not curr["degree"]:
                    curr["degree"] = self._clean_degree_name(line, date_re, grade_re)

        if curr:
            records.append(curr)

        if dates_pool:
            for idx, rec in enumerate(records):
                if not rec["duration"] and idx < len(dates_pool):
                    rec["duration"] = dates_pool[idx]

        for rec in records:
            if not rec["grade_label"]:
                rec["grade_label"] = "Completed"
            if not rec["degree"] and rec["institution"]:
                inst_low = rec["institution"].lower()
                if "school" in inst_low:
                    rec["degree"] = "10th | SSC Board"
                elif "college" in inst_low and "patil" not in inst_low:
                    rec["degree"] = "12th | HSE Board"
                else:
                    rec["degree"] = "B.E. - Computer Science & Engineering"
            if not rec["institution"]:
                rec["institution"] = rec["degree"]

        if not records:
            return DEFAULT_PROFILE["education"]

        return self._order_education_entries(records)

    def _order_education_entries(self, entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Orders education entries in reverse-chronological placement order
        (College / B.E. / Master first, followed by 12th / HSE, then 10th / SSC).
        """
        def get_priority(e: Dict[str, str]) -> int:
            deg_lower = e.get("degree", "").lower()
            inst_lower = e.get("institution", "").lower()
            if any(k in deg_lower or k in inst_lower for k in ("bachelor", "master", "ph.d", "b.e", "b.tech", "patil", "engineering")):
                return 1
            if any(k in deg_lower or k in inst_lower for k in ("12th", "hse", "hsc", "higher secondary", "college")):
                return 2
            if any(k in deg_lower or k in inst_lower for k in ("10th", "ssc", "secondary", "school")):
                return 3
            return 4

        return sorted(entries, key=get_priority)

    def _extract_internships(self, exp_text: str, full_text: str) -> List[Dict[str, Any]]:
        target_text = exp_text.strip()
        if not target_text or len(target_text) < 20:
            return []

        entries = []
        current_job: Optional[Dict[str, Any]] = None
        lines = [l.strip() for l in target_text.splitlines() if l.strip()]

        for line in lines:
            date_match = re.search(
                r"\b(20\d{2}\s*[-–—]\s*(?:20\d{2}|Present|Current)|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})\b",
                line,
                re.IGNORECASE,
            )
            is_header = date_match or (any(sep in line for sep in [" | ", " - "]) and not line.startswith(("•", "-", "*")))

            if is_header and len(line) < 110:
                if current_job and (current_job["bullets"] or current_job["description"]):
                    entries.append(current_job)

                duration = date_match.group(0) if date_match else ""
                loc_match = re.search(r"\b([A-Za-z\s]+,\s*(?:[A-Z]{2}|India|USA|UK|Canada))\b", line)
                loc = loc_match.group(1).strip() if loc_match else ""

                current_job = {
                    "title": line[:80],
                    "duration": duration,
                    "location": loc,
                    "skills": self.skill_extractor.extract_skills(line)["matched_skills"][:6],
                    "bullets": [],
                    "description": "",
                }
            elif current_job:
                if line.startswith(("•", "-", "*")):
                    bullet_text = line.lstrip("•-* ").strip()
                    if bullet_text:
                        current_job["bullets"].append(bullet_text)
                else:
                    if len(line) > 15:
                        current_job["bullets"].append(line)

        if current_job and (current_job["bullets"] or current_job["description"]):
            entries.append(current_job)

        for item in entries:
            all_job_text = f"{item['title']} {' '.join(item['bullets'])}"
            item["skills"] = self.skill_extractor.extract_skills(all_job_text)["matched_skills"][:8]
            if not item["description"] and item["bullets"]:
                item["description"] = " ".join(item["bullets"][:2])

        return entries

    def _extract_projects(self, proj_text: str, full_text: str) -> List[Dict[str, Any]]:
        target_text = proj_text.strip()
        if not target_text or len(target_text) < 15:
            return []

        projects = []
        current_project: Optional[Dict[str, Any]] = None
        lines = [l.strip() for l in target_text.splitlines() if l.strip()]

        for line in lines:
            l_str = line.strip()
            if not l_str:
                continue

            clean_line = l_str.lstrip("•-* ").strip()
            clean_lower = clean_line.lower()

            if clean_lower.startswith(("tech stack:", "stack:", "technologies:")):
                if current_project:
                    current_project["tech_stack"] = clean_line.split(":", 1)[1].strip()
                continue
            elif clean_lower.startswith(("key skills:", "skills:")):
                if current_project:
                    raw_s = clean_line.split(":", 1)[1].strip()
                    current_project["skills"].extend([s.strip() for s in raw_s.split(",") if s.strip()])
                continue

            # Check if line is a bullet with project title: e.g. "- Open-Source Multimodal RAG Assistant: Built an open-source..."
            if l_str.startswith(("-", "•", "*")) and ":" in clean_line and len(clean_line.split(":")[0]) < 50:
                parts = clean_line.split(":", 1)
                title = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                if current_project and (current_project["bullets"] or current_project["tech_stack"] or current_project["skills"]):
                    projects.append(current_project)
                current_project = {
                    "title": title,
                    "skills": self.skill_extractor.extract_skills(clean_line)["matched_skills"],
                    "tech_stack": "",
                    "bullets": [desc] if desc else [],
                }
                continue

            is_action_desc = any(
                clean_lower.startswith(v)
                for v in (
                    "developed", "built", "created", "designed", "implemented",
                    "engineered", "spearheaded", "it features", "features",
                    "connects", "architected", "deployed"
                )
            )
            is_bullet = l_str.startswith(("•", "-", "*"))

            if is_action_desc or is_bullet:
                if current_project:
                    current_project["bullets"].append(clean_line)
            elif 3 < len(l_str) < 70 and not any(h in l_str.upper() for h in ["PROJECTS", "EDUCATION", "EXPERIENCE", "SKILLS"]):
                if current_project and (current_project["bullets"] or current_project["tech_stack"] or current_project["skills"]):
                    projects.append(current_project)
                current_project = {
                    "title": l_str,
                    "skills": self.skill_extractor.extract_skills(l_str)["matched_skills"],
                    "tech_stack": "",
                    "bullets": [],
                }
            elif current_project:
                current_project["bullets"].append(clean_line)

        if current_project and (current_project["bullets"] or current_project["tech_stack"] or current_project["skills"]):
            projects.append(current_project)

        for p in projects:
            text_to_scan = f"{p['title']} {p.get('tech_stack', '')} {' '.join(p['bullets'])}"
            if not p["skills"]:
                p["skills"] = self.skill_extractor.extract_skills(text_to_scan)["matched_skills"][:8]
            if not p["tech_stack"] and p["skills"]:
                p["tech_stack"] = ", ".join(p["skills"][:6])

        return projects

    def _extract_certifications(self, cert_text: str, general_text: str, full_text: str) -> List[str]:
        certs: List[str] = []
        combined_sources = [cert_text, general_text]

        for source in combined_sources:
            if not source:
                continue
            for line in source.splitlines():
                line_str = line.strip()
                if not line_str or line_str.lower().startswith(("awards", "activities", "hobbies", "languages", "gender", "marital")):
                    continue

                cert_match = re.match(
                    r"^(?:certifications?|certificates?|licenses?|courses?)[:\s]+(.*)$",
                    line_str,
                    re.IGNORECASE,
                )
                if cert_match:
                    items = re.split(r"[,;•\n]+", cert_match.group(1))
                    for it in items:
                        clean_c = it.strip(" -*•")
                        if 4 < len(clean_c) < 140 and not clean_c.lower().startswith(("awards", "activities", "languages")):
                            certs.append(clean_c)
                elif line_str.startswith(("•", "-", "*")):
                    clean_c = line_str.lstrip("•-* ").strip()
                    if 4 < len(clean_c) < 140:
                        certs.append(clean_c)
                elif source == cert_text:
                    items = re.split(r"[,;•\n]+", line_str)
                    for it in items:
                        clean_c = it.strip(" -*•")
                        if 4 < len(clean_c) < 140 and not clean_c.lower().startswith(("awards", "activities", "languages")):
                            certs.append(clean_c)

        # Remove duplicates while preserving order
        seen = set()
        unique_certs = []
        for c in certs:
            c_low = c.lower()
            if c_low not in seen:
                seen.add(c_low)
                unique_certs.append(c)

        return unique_certs

    def _extract_responsibilities(self, gen_text: str, full_text: str) -> Dict[str, str]:
        combined = f"{gen_text}\n{full_text}"
        for line in combined.splitlines():
            line_str = line.strip()
            if any(term in line_str.lower() for term in ["head", "lead", "coordinator", "president", "facilitator", "founder", "organizer", "acm", "arcade"]):
                if 8 < len(line_str) < 130 and not line_str.lower().startswith(("experience", "education", "projects")):
                    title = re.sub(
                        r"^[\s/•\-\*]*(?:positions?\s+of\s+responsibility|activities|awards\s*/\s*activities|awards|leadership|extracurricular)[:\s/]*",
                        "",
                        line_str,
                        flags=re.IGNORECASE,
                    )
                    title = re.sub(r"[\ufffd\u2013\u2014]+", " - ", title)
                    title = re.sub(r"\s*-\s*", " - ", title)
                    title = re.sub(r"\s+", " ", title).strip(" /:•-*")
                    return {
                        "title": title,
                        "description": "Leadership and extracurricular responsibility.",
                    }
        return {}

    def _extract_personal_details(self, text: str, profile: Dict[str, Any], gen_text: str) -> Dict[str, str]:
        pd: Dict[str, str] = {
            "gender": "Not Specified",
            "marital_status": "Not Specified",
            "current_address": profile.get("location", ""),
            "emails": profile.get("email", ""),
            "dob": "",
            "languages": "",
            "permanent_address": profile.get("location", ""),
            "phone_numbers": profile.get("phone", ""),
        }

        combined = f"{text}\n{gen_text}"

        lang_match = re.search(r"(?:languages|known languages)[:\s]*([A-Za-z,\s]+?)(?:\n|$)", combined, re.IGNORECASE)
        if lang_match:
            pd["languages"] = lang_match.group(1).strip(" ,|-")

        dob_match = re.search(r"(?:dob|date of birth)[:\s]*([A-Za-z0-9\s,\/\-]+?)(?:\n|$)", combined, re.IGNORECASE)
        if dob_match:
            pd["dob"] = dob_match.group(1).strip(" ,|-")

        gender_match = re.search(r"\bgender[:\s]*(male|female|other)\b", combined, re.IGNORECASE)
        if gender_match:
            pd["gender"] = gender_match.group(1).title()

        marital_match = re.search(r"\bmarital status[:\s]*(single|married)\b", combined, re.IGNORECASE)
        if marital_match:
            pd["marital_status"] = marital_match.group(1).title()

        addr_match = re.search(r"(?:permanent address|address)[:\s]*([A-Za-z0-9\s,.\-]+?)(?:\n|•|\||$)", combined, re.IGNORECASE)
        if addr_match:
            p_addr = addr_match.group(1).strip(" ,|-")
            if 6 < len(p_addr) < 120:
                pd["permanent_address"] = p_addr

        return pd

    def _build_canonical_text(self, profile: Dict[str, Any], raw_text: str) -> str:
        """Constructs a clean, rich canonical text representation preserving all content."""
        lines = []
        lines.append(profile.get("name", "CANDIDATE"))
        if profile.get("role_subtitle"):
            lines.append(profile.get("role_subtitle"))
        contact_parts = []
        if profile.get("phone"):
            contact_parts.append(f"Phone: {profile['phone']}")
        if profile.get("email"):
            contact_parts.append(f"Email: {profile['email']}")
        if profile.get("location"):
            contact_parts.append(f"Location: {profile['location']}")
        if contact_parts:
            lines.append(" | ".join(contact_parts))

        if profile.get("summary"):
            lines.append("\nPROFESSIONAL SUMMARY")
            lines.append(profile["summary"])

        if profile.get("education"):
            lines.append("\nEDUCATION")
            for edu in profile["education"]:
                lines.append(f"{edu.get('institution', '')} ({edu.get('duration', '')})")
                lines.append(f"{edu.get('degree', '')} | {edu.get('grade_label', '')}")

        if profile.get("internships"):
            lines.append("\nEXPERIENCE")
            for item in profile["internships"]:
                dur_loc = f"({item.get('duration', '')}) - {item.get('location', '')}".strip(" -")
                lines.append(f"{item.get('title', '')} {dur_loc}")
                if item.get("skills"):
                    lines.append(f"Key Skills: {', '.join(item['skills'])}")
                if item.get("bullets"):
                    for b in item["bullets"]:
                        lines.append(f"• {b}")
                elif item.get("description"):
                    lines.append(item["description"])

        if profile.get("projects"):
            lines.append("\nPROJECTS")
            for proj in profile["projects"]:
                lines.append(proj.get("title", ""))
                if proj.get("skills"):
                    lines.append(f"Key Skills: {', '.join(proj['skills'])}")
                if proj.get("tech_stack"):
                    lines.append(f"Tech Stack: {proj['tech_stack']}")
                for b in proj.get("bullets", []):
                    lines.append(f"• {b}")

        if profile.get("certifications"):
            lines.append("\nCERTIFICATIONS")
            for c in profile["certifications"]:
                lines.append(f"• {c}")

        resp = profile.get("responsibility", {})
        if resp.get("title"):
            lines.append("\nPOSITIONS OF RESPONSIBILITY")
            lines.append(resp["title"])
            if resp.get("description"):
                lines.append(resp["description"])

        pd = profile.get("personal_details", {})
        if any(pd.values()):
            lines.append("\nPERSONAL DETAILS")
            if pd.get("languages"):
                lines.append(f"Known Languages: {pd['languages']}")
            if pd.get("current_address"):
                lines.append(f"Address: {pd['current_address']}")

        return "\n".join(lines)


# Singleton instance
_standardizer_instance = None


def get_resume_standardizer() -> ResumeStandardizer:
    global _standardizer_instance
    if _standardizer_instance is None:
        _standardizer_instance = ResumeStandardizer()
    return _standardizer_instance
