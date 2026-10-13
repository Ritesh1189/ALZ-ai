"""
Module: gap_analyzer.py
Performs deep skill gap analysis between candidate qualifications and target job roles.
Generates actionable upskilling roadmaps, hands-on portfolio project recommendations,
and milestone timelines to bridge technical gaps.
"""

from typing import Dict, List, Any, Optional


# Curated knowledge base of learning roadmaps for technical skills
SKILL_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "docker": {
        "priority": "Critical",
        "concepts": "Containerization, Dockerfile optimization, multi-stage builds, Docker Compose, volume mounts, container networking.",
        "project": "Containerize a full-stack Python or Node application with multi-stage builds, separate dev/prod configs, and Docker Compose orchestration.",
        "certifications": "Docker Certified Associate (DCA), freeCodeCamp Docker Course.",
        "estimated_weeks": 2,
    },
    "kubernetes": {
        "priority": "Critical",
        "concepts": "Pods, Deployments, Services, Ingress, ConfigMaps, Secrets, Horizontal Pod Autoscaling (HPA), Helm charts.",
        "project": "Deploy a microservice cluster using Minikube or Kind, set up rolling zero-downtime updates, and manage releases via Helm charts.",
        "certifications": "Certified Kubernetes Administrator (CKA), CKAD.",
        "estimated_weeks": 3,
    },
    "fastapi": {
        "priority": "Important",
        "concepts": "Asynchronous endpoints, Pydantic data schemas, dependency injection, OpenAPI documentation, background tasks, JWT authentication.",
        "project": "Build an asynchronous RESTful ML inference API serving a model with input validation, rate limiting, and automated Swagger documentation.",
        "certifications": "TestDriven.io FastAPI Mastery, FastAPI Official Tutorial.",
        "estimated_weeks": 1,
    },
    "aws": {
        "priority": "Critical",
        "concepts": "EC2, S3, RDS, Lambda serverless, IAM role least-privilege, VPC networking, CloudWatch, API Gateway.",
        "project": "Architect a serverless document processing pipeline using AWS Lambda triggered by S3 uploads, saving metadata into RDS/DynamoDB.",
        "certifications": "AWS Certified Solutions Architect Associate (SAA-C03), AWS Cloud Practitioner.",
        "estimated_weeks": 4,
    },
    "azure": {
        "priority": "Critical",
        "concepts": "Azure VMs, Blob Storage, Azure Functions, Entra ID, Virtual Networks, Azure DevOps pipelines.",
        "project": "Deploy a containerized web app to Azure App Service with automated GitHub Actions CI/CD and Azure Key Vault secret management.",
        "certifications": "Microsoft Certified: Azure Fundamentals (AZ-900), Azure Developer (AZ-204).",
        "estimated_weeks": 3,
    },
    "gcp": {
        "priority": "Critical",
        "concepts": "Compute Engine, Cloud Storage, Cloud Run, BigQuery, IAM policies, Cloud Build.",
        "project": "Deploy a containerized API to Google Cloud Run with autoscaling to zero and connect it to BigQuery for analytical logging.",
        "certifications": "Google Cloud Associate Cloud Engineer (ACE).",
        "estimated_weeks": 3,
    },
    "terraform": {
        "priority": "Important",
        "concepts": "Infrastructure as Code (IaC), providers, state management, remote backends (S3/GCS), reusable modules, variables and outputs.",
        "project": "Write modular Terraform code to provision a secure VPC, subnets, security groups, and an autoscaling compute cluster on AWS.",
        "certifications": "HashiCorp Certified: Terraform Associate.",
        "estimated_weeks": 2,
    },
    "ci/cd": {
        "priority": "Critical",
        "concepts": "Automated build matrices, unit/integration test triggers, secret management, artifact publishing, deployment environments.",
        "project": "Create a complete GitHub Actions CI/CD pipeline that runs linting, PyTest tests with code coverage, builds Docker images, and pushes to a registry.",
        "certifications": "GitHub Actions Certification, GitLab Certified CI/CD Specialist.",
        "estimated_weeks": 1,
    },
    "pytorch": {
        "priority": "Critical",
        "concepts": "Tensors, Autograd, custom nn.Module architectures, Dataset and DataLoader pipelines, GPU acceleration with CUDA, PyTorch Lightning.",
        "project": "Implement and train a custom vision or text classifier from scratch with custom training loops, checkpointing, and tensorboard logging.",
        "certifications": "DeepLearning.AI Deep Learning Specialization, Udacity PyTorch Scholarship.",
        "estimated_weeks": 4,
    },
    "tensorflow": {
        "priority": "Critical",
        "concepts": "Keras functional API, custom layers and loss functions, tf.data input pipelines, TensorBoard visualization, TF Serving.",
        "project": "Train a multi-input neural network with data augmentation pipelines, evaluate metrics, and export to TensorFlow SavedModel format.",
        "certifications": "TensorFlow Developer Certificate, DeepLearning.AI TF in Practice.",
        "estimated_weeks": 3,
    },
    "transformers": {
        "priority": "Critical",
        "concepts": "Self-attention mechanism, encoder-decoder architectures, Hugging Face AutoModel & AutoTokenizer, PEFT / LoRA fine-tuning.",
        "project": "Fine-tune a pretrained BERT or RoBERTa model on domain-specific dataset using Hugging Face Trainer and evaluate with precision-recall curves.",
        "certifications": "Hugging Face NLP Course, DeepLearning.AI Generative AI with LLMs.",
        "estimated_weeks": 2,
    },
    "llm": {
        "priority": "Critical",
        "concepts": "Tokenization, context windows, prompt design, temperature, top-p sampling, function calling, quantization (GGUF, AWQ).",
        "project": "Build an intelligent assistant capable of executing structured tool calls, validating outputs against JSON schemas, and streaming responses.",
        "certifications": "DeepLearning.AI ChatGPT Prompt Engineering & LangChain courses.",
        "estimated_weeks": 2,
    },
    "rag": {
        "priority": "Critical",
        "concepts": "Document chunking strategies, dense embeddings, vector indexing (HNSW, IVFFlat), hybrid semantic + BM25 search, reranking.",
        "project": "Build an end-to-end question-answering system over PDF documents with LangChain/LlamaIndex, ChromaDB vector store, and Cohere reranker.",
        "certifications": "DeepLearning.AI Building Systems with ChatGPT, LlamaIndex Mastery.",
        "estimated_weeks": 2,
    },
    "langchain": {
        "priority": "Important",
        "concepts": "LCEL (LangChain Expression Language), prompt templates, memory buffers, agent executors, tool binding, LangSmith observability.",
        "project": "Develop a multi-agent research assistant that searches the web, summarizes articles, and writes a synthesized report with citations.",
        "certifications": "LangChain Official Academy, DeepLearning.AI LangChain for LLM Apps.",
        "estimated_weeks": 2,
    },
    "vector databases": {
        "priority": "Important",
        "concepts": "Vector similarity metrics (Cosine, Euclidean, Dot Product), indexing algorithms, filtering metadata, Chroma, Pinecone, FAISS, Qdrant.",
        "project": "Benchmark FAISS vs ChromaDB on a 100,000 document embedding collection, evaluating query latency, memory consumption, and recall.",
        "certifications": "Pinecone Vector Academy, DeepLearning.AI Vector Databases.",
        "estimated_weeks": 1,
    },
    "mlops": {
        "priority": "Critical",
        "concepts": "Experiment tracking, model registry, artifact versioning, feature stores, data drift detection, continuous model retraining.",
        "project": "Set up an automated ML pipeline using MLflow for experiment logging, DVC for dataset versioning, and automated evaluation on new data pulls.",
        "certifications": "DeepLearning.AI MLOps Specialization (Andrew Ng), Databricks Certified ML Engineer.",
        "estimated_weeks": 3,
    },
    "scikit-learn": {
        "priority": "Critical",
        "concepts": "Classification, regression, cross-validation, Pipeline and ColumnTransformer, hyperparameter tuning with GridSearchCV.",
        "project": "Develop an end-to-end tabular prediction pipeline with automated preprocessing, outlier detection, and ensemble model stacking.",
        "certifications": "Kaggle Intermediate Machine Learning, Scikit-Learn MOOC (Inria).",
        "estimated_weeks": 2,
    },
    "sql": {
        "priority": "Critical",
        "concepts": "Window functions (ROW_NUMBER, RANK, LAG/LEAD), CTEs, complex joins, indexing strategies, query execution plans, aggregation.",
        "project": "Write advanced analytical SQL queries to solve multi-touch marketing attribution and monthly cohort retention analysis on sample business data.",
        "certifications": "Mode Analytics Advanced SQL, LeetCode SQL 50 Study Plan.",
        "estimated_weeks": 2,
    },
    "power bi": {
        "priority": "Important",
        "concepts": "Data modeling, star schema, DAX formulas (CALCULATE, RELATED, FILTER), power query M transformations, interactive dashboards.",
        "project": "Design an executive sales intelligence dashboard in Power BI featuring KPI cards, drill-through pages, and custom DAX metrics.",
        "certifications": "Microsoft Certified: Power BI Data Analyst Associate (PL-300).",
        "estimated_weeks": 2,
    },
    "tableau": {
        "priority": "Important",
        "concepts": "Calculated fields, Level of Detail (LOD) expressions, parameters, dashboard actions, storytelling with data visualization.",
        "project": "Create a public Tableau interactive storytelling dashboard tracking global supply chain metrics with dynamic filters and LOD calculations.",
        "certifications": "Tableau Certified Data Analyst.",
        "estimated_weeks": 2,
    },
    "react": {
        "priority": "Critical",
        "concepts": "Hooks (useState, useEffect, useMemo, useCallback), context API, state management, component lifecycle, virtual DOM, routing.",
        "project": "Build a responsive single-page web app with real-time state synchronization, custom hooks, and REST API integration.",
        "certifications": "Meta Front-End Developer Professional Certificate (Coursera), React Official Docs.",
        "estimated_weeks": 3,
    },
    "node.js": {
        "priority": "Critical",
        "concepts": "Event loop, asynchronous I/O, streams and buffers, Express middleware, authentication with JWT, error handling.",
        "project": "Build a scalable RESTful backend service with JWT authentication, rate limiting, and PostgreSQL integration.",
        "certifications": "OpenJS Node.js Application Developer (JSNAD), freeCodeCamp Backend Certificate.",
        "estimated_weeks": 3,
    },
    "opencv": {
        "priority": "Critical",
        "concepts": "Image transformations, color spaces, edge detection (Canny, Sobel), contour extraction, feature matching (ORB, SIFT), video streaming.",
        "project": "Build a real-time face and facial landmark tracking application operating over webcam feed at 30+ FPS.",
        "certifications": "OpenCV.org Computer Vision I & II.",
        "estimated_weeks": 2,
    },
    "yolo": {
        "priority": "Critical",
        "concepts": "Object detection bounding boxes, intersection over union (IoU), non-maximum suppression (NMS), YOLOv8 / Ultralytics API, custom dataset fine-tuning.",
        "project": "Train a custom YOLOv8 model on a labeled dataset (e.g., safety helmet detection or defect tracking) and benchmark mAP@0.5 score.",
        "certifications": "Ultralytics YOLO Docs, Roboflow Detection Certification.",
        "estimated_weeks": 2,
    },
    "cnn": {
        "priority": "Critical",
        "concepts": "Convolutional filters, pooling, stride, padding, receptive fields, transfer learning with ResNet / EfficientNet architectures.",
        "project": "Train a convolutional neural network with transfer learning on medical or satellite imagery, using Grad-CAM to visualize activation maps.",
        "certifications": "DeepLearning.AI Convolutional Neural Networks.",
        "estimated_weeks": 2,
    },
    "git": {
        "priority": "Critical",
        "concepts": "Branching strategies (Git Flow, Trunk-based), interactive rebase, cherry-pick, conflict resolution, merge vs rebase.",
        "project": "Set up a team Git repository with protected main branches, automated branch validation rules, and PR templates.",
        "certifications": "Pro Git Book, GitHub Foundations Certification.",
        "estimated_weeks": 1,
    },
}


class GapAnalyzer:
    """
    Analyzes missing skills and synthesizes an intelligent, structured upskilling path.
    """

    def analyze_gaps(
        self,
        target_role: Dict[str, Any],
        candidate_skills: List[str],
    ) -> Dict[str, Any]:
        """
        Analyzes the skill delta for a specific job role.
        """
        candidate_set = set(s.lower() for s in candidate_skills)
        required_list = target_role.get("required_skills", [])
        required_set = set(s.lower() for s in required_list)

        matched_skills = sorted(list(candidate_set.intersection(required_set)))
        missing_skills = sorted(list(required_set.difference(candidate_set)))

        missing_breakdown = []
        total_estimated_weeks = 0

        for skill in missing_skills:
            info = SKILL_KNOWLEDGE_BASE.get(skill, {
                "priority": "Recommended",
                "concepts": f"Core principles, best practices, and practical application of {skill}.",
                "project": f"Build a practical demonstration project showcasing end-to-end integration of {skill}.",
                "certifications": f"Official documentation and introductory course on {skill}.",
                "estimated_weeks": 2,
            })

            total_estimated_weeks += info["estimated_weeks"]
            missing_breakdown.append({
                "skill": skill,
                "priority": info["priority"],
                "concepts": info["concepts"],
                "project": info["project"],
                "certifications": info["certifications"],
                "estimated_weeks": info["estimated_weeks"],
            })

        # Sort missing skills: Critical first, then Important, then Recommended
        priority_order = {"Critical": 1, "Important": 2, "Recommended": 3}
        missing_breakdown.sort(key=lambda x: priority_order.get(x["priority"], 4))

        # Generate Milestone Roadmap
        roadmap_milestones = self._build_roadmap_phases(missing_breakdown)

        coverage_ratio = (
            round((len(matched_skills) / len(required_set)) * 100, 1)
            if required_set
            else 100.0
        )

        return {
            "role_name": target_role.get("role_name", "Unknown Role"),
            "category": target_role.get("category", "General"),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_count": len(matched_skills),
            "missing_count": len(missing_skills),
            "total_required": len(required_set),
            "coverage_pct": coverage_ratio,
            "missing_skill_details": missing_breakdown,
            "estimated_weeks_total": total_estimated_weeks,
            "roadmap_phases": roadmap_milestones,
            "readiness_summary": self._generate_readiness_summary(coverage_ratio, len(missing_skills)),
        }

    def _build_roadmap_phases(self, missing_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organizes missing skills into 3 phased learning milestones.
        """
        if not missing_items:
            return [{
                "phase": "Phase 1: Portfolio Polish & Application",
                "duration": "1 - 2 Weeks",
                "objective": "All core skills matched! Polish your resume and portfolio projects.",
                "skills": [],
                "deliverable": "Tailor resume bullets to the target company's specific job description.",
            }]

        critical_skills = [i for i in missing_items if i["priority"] == "Critical"]
        other_skills = [i for i in missing_items if i["priority"] != "Critical"]

        phases = []

        # Phase 1: High Priority Foundations
        p1_skills = critical_skills[:3] if critical_skills else other_skills[:2]
        p1_names = [s["skill"] for s in p1_skills]
        p1_weeks = sum(s["estimated_weeks"] for s in p1_skills) or 2
        phases.append({
            "phase": "Phase 1: Core Foundation & Tooling",
            "duration": f"Weeks 1 - {min(4, p1_weeks)}",
            "objective": f"Master fundamental concepts and setup for {', '.join(p1_names)}.",
            "skills": p1_names,
            "deliverable": f"Complete foundational hands-on exercises and implement {p1_names[0] if p1_names else 'core stack'} in local dev environment.",
        })

        # Phase 2: Implementation & Systems
        p2_skills = critical_skills[3:] + other_skills[:2]
        if p2_skills:
            p2_names = [s["skill"] for s in p2_skills]
            phases.append({
                "phase": "Phase 2: End-to-End Project Development",
                "duration": "Weeks 4 - 6",
                "objective": f"Synthesize {', '.join(p2_names)} into a real-world deployed project.",
                "skills": p2_names,
                "deliverable": "Publish a comprehensive GitHub repository featuring clean architecture, unit tests, and live demo link.",
            })

        # Phase 3: Cloud, Optimization & Interview Readiness
        rem_skills = other_skills[2:]
        rem_names = [s["skill"] for s in rem_skills]
        phases.append({
            "phase": "Phase 3: Production Hardening & Interview Prep",
            "duration": "Weeks 7 - 8",
            "objective": "System design deep dive, behavioral storytelling, and technical interview simulations.",
            "skills": rem_names,
            "deliverable": "Add quantified bullet points highlighting the new projects to your resume.",
        })

        return phases

    def _generate_readiness_summary(self, coverage_pct: float, missing_count: int) -> str:
        if coverage_pct >= 80.0:
            return f"You are well-aligned for this position ({coverage_pct}% skill match). Addressing {missing_count} minor skill gaps will make your application stand out."
        elif coverage_pct >= 55.0:
            return f"You have a solid foundation with {coverage_pct}% skill match. Completing the recommended 4 to 6 week roadmap will position you as a competitive candidate."
        else:
            return f"This role requires significant upskilling ({coverage_pct}% match, {missing_count} skills missing). Follow the phased roadmap to systematically build the required competencies."


# Singleton
_default_gap_analyzer: GapAnalyzer = GapAnalyzer()


def get_gap_analyzer() -> GapAnalyzer:
    global _default_gap_analyzer
    return _default_gap_analyzer
