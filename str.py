import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Function to extract text from PDF resume
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Smart Job Recommender", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f4f4f4;}
    .sidebar .sidebar-content {background-color: #2C3E50; color: white;}
    .stButton>button {border-radius: 8px; background-color: #e74c3c; color: white;}
    .stButton>button:hover {background-color: #c0392b;}
    .stTextInput>div>div>input {border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Smart Job Recommender</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-size: 18px;'>Upload your resume and job listings to find your best job matches based on skills!</p>", unsafe_allow_html=True)

# Upload section
st.markdown("### Upload Your Files")
col1, col2 = st.columns(2)

with col1:
    jobs_file = st.file_uploader("Upload LinkedIn Jobs CSV", type=["csv"])

with col2:
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

if jobs_file and resume_file:
    job_data = pd.read_csv(jobs_file)
    resume_text = extract_text_from_pdf(resume_file)
    
    # Apply preprocessing
    job_data["Processed_Skills"] = job_data["Skills"].fillna('').apply(preprocess_text)
    resume_skills = preprocess_text(resume_text)
    
    # TF-IDF Calculation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_skills] + job_data["Processed_Skills"].tolist())
    skill_similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # BM25 Algorithm
    tokenized_jobs = [skills.split() for skills in job_data["Processed_Skills"]]
    bm25 = BM25Okapi(tokenized_jobs)
    resume_tokens = resume_skills.split()
    bm25_scores = bm25.get_scores(resume_tokens)
    
    # Create Graph for PageRank
    G = nx.DiGraph()
    num_jobs = len(job_data)
    
    for i in range(num_jobs + 1):
        G.add_node(i)
    
    for i in range(1, num_jobs + 1):
        skill_weight = skill_similarity_matrix[0, i] * 10
        bm25_weight = bm25_scores[i - 1] * 5
        final_weight = skill_weight + bm25_weight
        
        if final_weight > 0.1:
            G.add_edge(0, i, weight=final_weight)
    
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    final_scores = {i: pagerank_scores[i] * 0.6 + (bm25_scores[i - 1] / max(bm25_scores)) * 0.4 for i in range(1, num_jobs + 1)}
    sorted_jobs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Display top matches
    st.markdown("### Top Job Matches for You")
    for rank, (index, score) in enumerate(sorted_jobs[:5], start=1):
        job = job_data.iloc[index - 1]
        with st.expander(f"🎯 {rank}. {job['Title']} at {job['Company']} ({job['Location']}) - Score: {score:.4f}"):
            st.write(f"**Experience Required:** {job['Experience']}")
            st.write(f"**Skills:** {job['Skills']}")
            st.markdown(f"[👉 Apply Here]({job['Job URL']})", unsafe_allow_html=True)
