"""
Resume Formatter Module - D.Y. Patil College / Professional ATS Placement Template
Generates pixel-perfect, print-to-PDF ready HTML resumes matching the official format.
"""

from typing import Dict, Any, Optional


DEFAULT_PROFILE = {
    "name": "RITESH NATHARAM TURE",
    "role_subtitle": "B.E. - Computer Science & Engineering",
    "phone": "+91-8983497601",
    "email": "tureritesh413@gmail.com",
    "location": "Pune, Maharashtra, India - 411033",
    "college_logo_title": "DYP",
    "college_subtext": "DR. D. Y. PATIL PRATISHTHAN",
    "education": [
        {
            "institution": "D.Y. Patil College of Engineering, Akurdi",
            "duration": "2023 - 2027",
            "degree": "B.E. - Computer Science & Engineering",
            "grade_label": "CGPA: 7.80 / 10",
        },
        {
            "institution": "Late Rangnathrao Kaldate Guruji higher and secondary college parbhani, Parbhani",
            "duration": "2023",
            "degree": "12th | MSBSHSE",
            "grade_label": "Percentage: 68.33 / 100",
        },
        {
            "institution": "Aurbindo Aksharjyoti School Parbhani, Parbhani",
            "duration": "2021",
            "degree": "10th | MSBSHSE",
            "grade_label": "Percentage: 82 / 100",
        },
    ],
    "internships": [
        {
            "title": "The Develepors Arena | IT / Computers - Software",
            "duration": "10 Jan, 2026 - 10 Feb, 2026",
            "location": "Pune",
            "skills": ["Python", "MySQL", "Pandas", "CRUD Operations", "Data Management", "File Handling"],
            "description": "Developed a Contact Management System using Python to manage contact information efficiently. Implemented core functionalities to add, view, search, update, and delete contact records. Applied Python programming concepts, data handling, and problem-solving skills to build a functional application. Improved practical software development skills through hands-on project implementation and testing.",
        }
    ],
    "projects": [
        {
            "title": "AI Resume Analyzer and Job Recommendation System",
            "skills": ["Python", "NLP", "Scikit-Learn", "TF-IDF", "Cosine Similarity", "Pandas", "Streamlit", "Recommender Systems"],
            "tech_stack": "Python, Streamlit, NLP, Scikit-learn, TF-IDF, Cosine Similarity, Pandas, NumPy, pypdf, python-docx, CSV",
            "bullets": [
                "Developing an NLP-based application using Python, Scikit-learn, and Streamlit to analyze resumes, extract technical skills, calculate job-role matching scores using TF-IDF and cosine similarity, recommend suitable job roles, identify skill gaps, and generate personalized learning roadmaps."
            ],
        },
        {
            "title": "Skill Match Job Portal",
            "skills": ["React.js", "Spring Boot", "MongoDB Atlas", "Swagger UI", "REST APIs"],
            "tech_stack": "React.js, Spring Boot, MongoDB Atlas, Swagger UI, REST APIs",
            "bullets": [
                "Built a full-stack job portal using React.js, Spring Boot, and MongoDB Atlas to match job seekers with recruiters based on skills, reducing job search time by 40%.",
                "This job portal connects job seekers and recruiters through skill-based job matching. Built an interactive frontend using React.js and developed backend REST APIs using Spring Boot. Integrated MongoDB Atlas for data storage and Swagger UI for API documentation and testing."
            ],
        },
    ],
    "responsibility": {
        "title": "ACM Documentation and Content Creator Head",
        "description": "As the Documentation and Content Creator Head of ACM, I lead the creation, management, and presentation of all official content and documentation. I ensure accurate event records, engaging digital content, and consistent branding across platforms while coordinating with different teams to effectively showcase ACM's activities and achievements. I strategize content planning and storytelling to effectively communicate ACM's vision, initiatives, and technical impact to a wider audience."
    },
    "personal_details": {
        "gender": "Male",
        "marital_status": "Single",
        "current_address": "gurudwara chowk ,akurdi railway station pune, gurudwara chowk ,akurdi railway station pune, Pune, Maharashtra, India - 411033",
        "emails": "tureritesh413@gmail.com , riteshture089@gmail.com",
        "dob": "06 Nov, 2005",
        "languages": "English, Marathi, Hindi",
        "permanent_address": "AT. SIMANGAON TQ. SELU DIST PARBHANI, bori parbhani, Parbhani, Maharashtra, India - 431401",
        "phone_numbers": "+91-8983497601, +91-9657447301",
    }
}


def render_dyp_resume_html(profile_data: Optional[Dict[str, Any]] = None, enable_toolbar: bool = True) -> str:
    """
    Renders an HTML string for the D.Y. Patil placement resume format.
    """
    data = dict(DEFAULT_PROFILE)
    if profile_data:
        data.update(profile_data)

    # Render summary if present
    summary_html = ""
    if data.get("summary") and data.get("summary").strip():
        summary_html = f"""
        <div class="section-banner">PROFESSIONAL SUMMARY</div>
        <div style="padding: 8px 16px 10px 16px; font-size: 11px; color: #334155; line-height: 1.45;">
          {data.get('summary', '').strip()}
        </div>
        """

    # Render education items
    edu_html = ""
    if data.get("education"):
        edu_entries_html = ""
        for edu in data.get("education", []):
            degree_clean = edu.get("degree", "").replace("12th", "12<sup>th</sup>").replace("10th", "10<sup>th</sup>")
            grade_part = f" &nbsp;|&nbsp; <strong>{edu.get('grade_label')}</strong>" if edu.get("grade_label") else ""
            edu_entries_html += f"""
            <div class="edu-entry">
              <div class="row-flex">
                <div class="institution-name">{edu.get('institution', '')}</div>
                <div class="date-badge">{edu.get('duration', '')}</div>
              </div>
              <div class="edu-details">
                {degree_clean}{grade_part}
              </div>
            </div>
            """
        edu_html = f"""
        <div class="section-banner">EDUCATION</div>
        {edu_entries_html}
        """

    # Render internships / work experience
    intern_html = ""
    if data.get("internships"):
        items_html = ""
        for item in data.get("internships", []):
            skills_pills = "".join([f'<span class="skill-pill">{s}</span>' for s in item.get("skills", [])])
            bullets_li = "".join([f'<li>{b}</li>' for b in item.get("bullets", [])])
            bullets_html = f'<ul class="bullet-list">{bullets_li}</ul>' if bullets_li else f'<p class="desc-text">{item.get("description", "")}</p>'
            
            items_html += f"""
            <div class="item-block">
              <div class="row-flex">
                <div class="item-title">{item.get('title', '')}</div>
                <div class="date-badge">{item.get('duration', '')}</div>
              </div>
              <div class="item-subtitle">{item.get('location', '')}</div>
              <div class="skills-row">
                <span class="skills-label">Key Skills:</span>
                {skills_pills}
              </div>
              {bullets_html}
            </div>
            """
        intern_html = f"""
        <div class="section-banner">EXPERIENCE & INTERNSHIPS</div>
        {items_html}
        """

    # Render projects
    proj_html = ""
    if data.get("projects"):
        proj_items_html = ""
        for proj in data.get("projects", []):
            skills_pills = "".join([f'<span class="skill-pill">{s}</span>' for s in proj.get("skills", [])])
            bullets_li = ""
            if proj.get("tech_stack"):
                bullets_li += f"<li><strong>Tech Stack:</strong> {proj.get('tech_stack')}</li>"
            for b in proj.get("bullets", []):
                bullets_li += f"<li>{b}</li>"

            proj_items_html += f"""
            <div class="item-block">
              <div class="item-title">{proj.get('title', '')}</div>
              <div class="skills-row">
                <span class="skills-label">Key Skills:</span>
                {skills_pills}
              </div>
              <ul class="bullet-list">
                {bullets_li}
              </ul>
            </div>
            """
        proj_html = f"""
        <div class="section-banner">PROJECTS</div>
        {proj_items_html}
        """

    # Render certifications if present
    certs_html = ""
    if data.get("certifications"):
        certs_li = "".join([f"<li>{c}</li>" for c in data.get("certifications", [])])
        certs_html = f"""
        <div class="section-banner">CERTIFICATIONS</div>
        <div style="padding: 6px 16px 10px 16px;">
          <ul class="bullet-list">
            {certs_li}
          </ul>
        </div>
        """

    # Render responsibilities
    resp = data.get("responsibility", {})
    resp_html = ""
    if resp.get("title") or resp.get("description"):
        resp_html = f"""
        <div class="section-banner">POSITIONS OF RESPONSIBILITY</div>
        <div class="responsibility-block">
          <div class="resp-title">{resp.get('title', '')}</div>
          <p class="desc-text">{resp.get('description', '')}</p>
        </div>
        """

    # Personal details
    pd = data.get("personal_details", {})

    toolbar_html = ""
    if enable_toolbar:
        toolbar_html = """
        <div class="toolbar no-print">
          <div class="toolbar-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            <span>Placement Resume Format & Preview</span>
          </div>
          <div class="toolbar-actions">
            <button class="btn btn-secondary" onclick="toggleEditMode()" id="editBtn">
              Edit Content
            </button>
            <button class="btn" onclick="window.print()">
              Print / Save as PDF (Ctrl + P)
            </button>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{data.get('name', 'Resume')} - D.Y. Patil Resume Format</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap" rel="stylesheet">
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Roboto', Arial, sans-serif;
      color: #1a202c;
      background-color: #f8fafc;
      line-height: 1.35;
      font-size: 11px;
    }}

    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 1000;
      background: #0f172a;
      color: #ffffff;
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    .toolbar-title {{
      font-weight: 700;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .toolbar-actions {{
      display: flex;
      gap: 8px;
    }}

    .btn {{
      background: #0088cc;
      color: #ffffff;
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 11.5px;
      cursor: pointer;
      transition: all 0.2s;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .btn:hover {{
      background: #0077b5;
    }}

    .btn-secondary {{
      background: #334155;
    }}

    .btn-secondary:hover {{
      background: #475569;
    }}

    .resume-page {{
      width: 210mm;
      min-height: 297mm;
      margin: 16px auto;
      background: #ffffff;
      padding: 12mm 15mm 10mm 15mm;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      position: relative;
    }}

    .header-table {{
      width: 100%;
      margin-bottom: 6px;
      border-collapse: collapse;
    }}

    .header-logo-col {{
      width: 130px;
      vertical-align: top;
    }}

    .dyp-logo-card {{
      background: #000000;
      color: #ffffff;
      width: 125px;
      height: 68px;
      border-radius: 3px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      padding: 6px 8px;
    }}

    .dyp-text {{
      font-size: 26px;
      font-weight: 900;
      letter-spacing: 1.5px;
      line-height: 1;
      font-family: 'Arial Black', Arial, sans-serif;
      color: #ffffff;
    }}

    .dyp-text span {{
      color: #0088cc;
    }}

    .dyp-subtext {{
      font-size: 5.5px;
      font-weight: 700;
      letter-spacing: 0.8px;
      color: #bfa15f;
      text-transform: uppercase;
      margin-top: 5px;
      line-height: 1.1;
    }}

    .header-info-col {{
      text-align: right;
      vertical-align: top;
    }}

    .candidate-name {{
      font-size: 20px;
      font-weight: 800;
      color: #0088cc;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 2px;
    }}

    .candidate-role {{
      font-size: 11.5px;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 2px;
    }}

    .candidate-meta {{
      font-size: 10.5px;
      color: #334155;
      line-height: 1.35;
    }}

    .candidate-meta a {{
      color: #0088cc;
      text-decoration: none;
    }}

    .section-banner {{
      background-color: #0088cc;
      color: #ffffff;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 3.5px 8px;
      margin-top: 8px;
      margin-bottom: 6px;
      border-radius: 2px;
    }}

    .edu-entry {{
      margin-bottom: 6px;
    }}

    .row-flex {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}

    .institution-name {{
      font-size: 11px;
      font-weight: 700;
      color: #0f172a;
    }}

    .date-badge {{
      font-size: 10.5px;
      font-weight: 600;
      color: #1e293b;
      white-space: nowrap;
    }}

    .edu-details {{
      font-size: 10.5px;
      color: #334155;
      margin-top: 1px;
    }}

    .edu-details sup {{
      font-size: 7.5px;
    }}

    .item-block {{
      margin-bottom: 6px;
    }}

    .item-title {{
      font-size: 11px;
      font-weight: 700;
      color: #0f172a;
    }}

    .item-subtitle {{
      font-size: 10.5px;
      color: #475569;
      margin-bottom: 2px;
    }}

    .skills-row {{
      margin: 2px 0 4px 0;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 3px;
    }}

    .skills-label {{
      font-size: 10.5px;
      font-weight: 700;
      color: #0f172a;
      margin-right: 2px;
    }}

    .skill-pill {{
      display: inline-block;
      background-color: #f1f5f9;
      color: #1e293b;
      border: 1px solid #cbd5e1;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 9.8px;
      font-weight: 500;
      line-height: 1.25;
    }}

    .desc-text {{
      font-size: 10.2px;
      color: #27272a;
      line-height: 1.35;
      text-align: justify;
    }}

    .bullet-list {{
      list-style-type: disc;
      padding-left: 14px;
      margin-top: 2px;
    }}

    .bullet-list li {{
      font-size: 10.2px;
      color: #27272a;
      line-height: 1.35;
      margin-bottom: 2px;
      text-align: justify;
    }}

    .bullet-list li strong {{
      font-weight: 700;
      color: #0f172a;
    }}

    .responsibility-block {{
      margin-bottom: 6px;
    }}

    .resp-title {{
      font-size: 11px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 2px;
    }}

    .details-grid {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 2px;
    }}

    .details-grid td {{
      vertical-align: top;
      padding: 1.5px 0;
      font-size: 10.2px;
      line-height: 1.35;
    }}

    .details-grid .col-left {{
      width: 50%;
      padding-right: 12px;
    }}

    .details-grid .col-right {{
      width: 50%;
      padding-left: 6px;
    }}

    .field-label {{
      font-weight: 700;
      color: #0f172a;
    }}

    .field-val {{
      color: #27272a;
    }}

    .field-val a {{
      color: #0088cc;
      text-decoration: none;
    }}

    @media print {{
      body {{
        background: #ffffff !important;
        font-size: 10.5px;
      }}

      .no-print, .toolbar {{
        display: none !important;
      }}

      .resume-page {{
        width: 100% !important;
        min-height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
      }}

      .section-banner {{
        background-color: #0088cc !important;
        color: #ffffff !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }}

      .dyp-logo-card {{
        background: #000000 !important;
        color: #ffffff !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }}

      .skill-pill {{
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
      }}

      @page {{
        size: A4 portrait;
        margin: 8mm 12mm 8mm 12mm;
      }}
    }}
  </style>
</head>
<body>

  {toolbar_html}

  <div class="resume-page" id="resumeContent">
    
    <!-- HEADER -->
    <table class="header-table">
      <tr>
        <td class="header-logo-col">
          <div class="dyp-logo-card">
            <div class="dyp-text">{data.get('college_logo_title', 'DYP')}</div>
            <div class="dyp-subtext">{data.get('college_subtext', 'DR. D. Y. PATIL PRATISHTHAN')}</div>
          </div>
        </td>
        <td class="header-info-col">
          <div class="candidate-name">{data.get('name', '')}</div>
          <div class="candidate-role">{data.get('role_subtitle', '')}</div>
          <div class="candidate-meta">Ph: {data.get('phone', '')}</div>
          <div class="candidate-meta">Email: <a href="mailto:{data.get('email', '')}">{data.get('email', '')}</a></div>
          <div class="candidate-meta">{data.get('location', '')}</div>
        </td>
      </tr>
    </table>

    {summary_html}

    {edu_html}

    {intern_html}

    {proj_html}

    {certs_html}

    {resp_html}

    <!-- PERSONAL DETAILS -->
    <div class="section-banner">PERSONAL DETAILS</div>

    <table class="details-grid">
      <tr>
        <td class="col-left">
          <div><span class="field-label">Gender:</span> <span class="field-val">{pd.get('gender', 'Male')}</span></div>
          <div><span class="field-label">Marital Status:</span> <span class="field-val">{pd.get('marital_status', 'Single')}</span></div>
          <div>
            <span class="field-label">Current Address:</span> 
            <span class="field-val">{pd.get('current_address', '')}</span>
          </div>
          <div style="margin-top:2px;">
            <span class="field-label">Emails:</span> 
            <span class="field-val">{pd.get('emails', '')}</span>
          </div>
        </td>
        <td class="col-right">
          <div><span class="field-label">Date of Birth:</span> <span class="field-val">{pd.get('dob', '')}</span></div>
          <div><span class="field-label">Known Languages:</span> <span class="field-val">{pd.get('languages', '')}</span></div>
          <div>
            <span class="field-label">Permanent Address:</span> 
            <span class="field-val">{pd.get('permanent_address', '')}</span>
          </div>
          <div style="margin-top:2px;">
            <span class="field-label">Phone Numbers:</span> 
            <span class="field-val">{pd.get('phone_numbers', '')}</span>
          </div>
        </td>
      </tr>
    </table>

  </div>

  <script>
    let isEditing = false;
    function toggleEditMode() {{
      const page = document.getElementById('resumeContent');
      const btn = document.getElementById('editBtn');
      isEditing = !isEditing;
      page.contentEditable = isEditing;
      if (isEditing) {{
        btn.innerText = 'Save / Lock Edits';
        btn.style.background = '#10b981';
      }} else {{
        btn.innerText = 'Edit Content';
        btn.style.background = '#334155';
      }}
    }}
  </script>
</body>
</html>
"""
    return html
