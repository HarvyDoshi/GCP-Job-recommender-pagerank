import pandas as pd
import numpy as np
import networkx as nx
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  # BM25 Library
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Function to extract text from PDF resume
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Load job dataset
job_data = pd.read_csv("linkedin_ml_jobs_India.csv")

# Extract resume text from PDF
resume_text = extract_text_from_pdf("Harvy_Doshi_Resume (21).pdf")

# Preprocessing function (Stopword Removal + Stemming)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
job_data["Processed_Skills"] = job_data["Skills"].fillna('').apply(preprocess_text)
resume_skills = preprocess_text(resume_text)

# --- TF-IDF for Skills ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([resume_skills] + job_data["Processed_Skills"].tolist())

# Compute cosine similarity matrix for SKILLS
skill_similarity_matrix = cosine_similarity(tfidf_matrix)

# --- BM25 Algorithm for Matching Jobs ---
tokenized_jobs = [skills.split() for skills in job_data["Processed_Skills"]]
bm25 = BM25Okapi(tokenized_jobs)
resume_tokens = resume_skills.split()
bm25_scores = bm25.get_scores(resume_tokens)

# --- Create Graph for PageRank ---
G = nx.DiGraph()
num_jobs = len(job_data)

# Resume experience (Extract from resume manually if structured)
resume_experience = 3  # Adjust if known

# Add nodes
for i in range(num_jobs + 1):
    G.add_node(i)

# Add edges: Resume (0) to all jobs
for i in range(1, num_jobs + 1):
    job_experience = job_data.iloc[i - 1]["Experience"]
    job_experience = int(re.search(r'\d+', str(job_experience)).group(0)) if re.search(r'\d+', str(job_experience)) else 0

    # Compute hybrid weight
    skill_weight = skill_similarity_matrix[0, i] * 10  # More weight for skills
    experience_weight = 0.1 * (1 - abs(resume_experience - job_experience) / 10)  # Lower impact
    bm25_weight = bm25_scores[i - 1] * 5  # Boost BM25 relevance

    final_weight = skill_weight + experience_weight + bm25_weight

    if final_weight > 0.1:
        G.add_edge(0, i, weight=final_weight)

# Compute PageRank
pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")

# Combine BM25 and PageRank scores
final_scores = {i: pagerank_scores[i] * 0.6 + (bm25_scores[i - 1] / max(bm25_scores)) * 0.4 for i in range(1, num_jobs + 1)}

# Rank jobs
sorted_jobs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

# Display results
print("\nTop 5 Job Matches:\n")
for rank, (index, score) in enumerate(sorted_jobs[:5], start=1):  
    job = job_data.iloc[index - 1]  
    print(f"{rank}. {job['Title']} at {job['Company']} ({job['Location']})")
    print(f"   Experience: {job['Experience']}")
    print(f"   Skills: {job['Skills']}")
    print(f"   Job URL: {job['Job URL']}")
    print(f"   Final Score: {score:.4f}")
    print("-" * 50)
