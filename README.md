# GCP-Job-recommender-pagerank
Job Recommendation Engine with PageRank algorithm on GCP
A smart job matching system that ranks LinkedIn job postings using PageRank algorithms, deployed on Google Kubernetes Engine (GKE).

🚀 Key Features
Web Scraping: Crawls LinkedIn for job postings (titles, skills, experience) using Python/BeautifulSoup.

PageRank Algorithm: Ranks jobs based on skill similarity, company relevance, and experience matching using networkx.

Resume-Job Matching:

TF-IDF/BM25: Analyzes resume skills vs. job descriptions.

Hybrid Scoring: Combines PageRank and cosine similarity for personalized recommendations.

GCP Deployment: Containerized with Docker and deployed on GKE with LoadBalancer for scalability.

🛠️ Tech Stack
Backend: Python (Pandas, NetworkX, Scikit-learn, BeautifulSoup)

NLP: NLTK, TF-IDF, BM25, cosine similarity

Infrastructure: Docker, Kubernetes (GKE), Google Cloud Load Balancing

Data: CSV/PDF processing (PyMuPDF)

📂 Repository Structure
Copy
/  
├── web scraper/           # LinkedIn job scraper (code.py)  
├── pagerank/              # Job ranking with PageRank (code1pg_rank.py)  
├── resume_matcher/        # Resume-job matching (code2_resume.py)  
├── Dockerfile             # Containerization for GKE  
├── requirements.txt       # Python dependencies  
└── kubernetes/            # GKE deployment manifests (YAML)  
⚡ Quick Start
Scrape Jobs:

bash
Copy
python code.py  # Output: linkedin_ml_jobs_India.csv
Rank Jobs:

bash
Copy
python code1pg_rank.py  # Generates ranked_jobs.csv
Match Resume:

bash
Copy
python code2_resume.py Harvy_Doshi_Resume.pdf
Deploy on GKE:

bash
Copy
kubectl apply -f kubernetes/deployment.yaml
🌐 Live Demo

📌 Why This Project?
Solves cold-start problems in job searches by ranking less visible but relevant postings.

Hybrid approach (PageRank + NLP) outperforms keyword-only matching.

Scalable via GKE—handles 1000s of job listings.

🎯 Perfect for:

Job seekers wanting AI-powered recommendations

Recruiters analyzing skill-demand trends

Learners interested in NLP + Graph Theory applications
