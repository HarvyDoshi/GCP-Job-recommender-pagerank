import requests
from bs4 import BeautifulSoup
import math
import pandas as pd
import time
import random
import re

# List to store job IDs
job_ids = []

# List to store all job details
all_jobs = []

# Headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}

# Search parameters
keywords = "Machine learning"  # SDE jobs
location = "India"

# Encode the search parameters for the URL
encoded_keywords = keywords.replace(" ", "%20")
encoded_location = location.replace(" ", "%20")

# URL for the job search
target_url = f'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={encoded_keywords}&location={encoded_location}&start={{}}'

# Estimate the number of jobs to scrape (adjust as needed)
estimated_job_count = 400

# Calculate the number of pages to scrape
pages_to_scrape = math.ceil(estimated_job_count / 25)

print(f"Searching for {keywords} jobs in {location}...")
print(f"Will scrape approximately {pages_to_scrape} pages of results")

# Scrape job IDs from search results
for i in range(0, pages_to_scrape):
    try:
        print(f"Scraping page {i+1}/{pages_to_scrape}")
        
        # Add a random delay to avoid getting blocked
        time.sleep(random.uniform(1, 3))
        
        # Send the request to LinkedIn
        res = requests.get(target_url.format(i*25), headers=headers)
        
        # Check if the request was successful
        if res.status_code != 200:
            print(f"Failed to retrieve page {i+1}. Status code: {res.status_code}")
            continue
            
        # Parse the HTML content
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Find all job listings on the page
        alljobs_on_this_page = soup.find_all("li")
        
        print(f"Found {len(alljobs_on_this_page)} jobs on this page")
        
        # If no jobs found, break the loop
        if len(alljobs_on_this_page) == 0:
            print("No more jobs found. Stopping search.")
            break
            
        # Extract job IDs from each job listing
        for x in range(0, len(alljobs_on_this_page)):
            try:
                job_card = alljobs_on_this_page[x].find("div", {"class": "base-card"})
                if job_card and job_card.get('data-entity-urn'):
                    jobid = job_card.get('data-entity-urn').split(":")[-1]
                    job_ids.append(jobid)
            except Exception as e:
                print(f"Error extracting job ID: {e}")
                continue
                
    except Exception as e:
        print(f"Error scraping page {i+1}: {e}")
        continue

print(f"Total job IDs collected: {len(job_ids)}")

# URL for individual job postings
job_url = 'https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}'

# Function to extract experience from job description
def extract_experience(description):
    experience_patterns = [
        r"(\d+\+?\s*-?\s*\d*\+?\s*years?).{0,30}experience",
        r"experience.{0,30}(\d+\+?\s*-?\s*\d*\+?\s*years?)",
        r"(\d+\+?\s*-?\s*\d*\+?\s*years?)",
        r"(\d+\+?\s*to\s*\d+\+?\s*years?).{0,30}experience",
        r"(\d+\+?\s*years?).{0,30}experience",
        r"experience.{0,30}(\d+\+?\s*years?)",
        r"experience.{0,30}(\d+-\d+\s*years?)",
        r"experience.{0,30}(\d+\+?\s*to\s*\d+\+?\s*years?)",
        r"experience.{0,10}(\d+\+?)",
        r"(\d+\+?).{0,10}experience"
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Not specified"

# Function to extract skills from job description
def extract_skills(description):
    # List of AI, ML, and Data Science-related skills to look for
    ai_ml_skills = [
        "machine learning", "artificial intelligence", "deep learning", "data science", "big data", "hadoop", "spark", "tensorflow",
        "pytorch", "scikit-learn", "nlp", "natural language processing", "computer vision", "opencv", "reinforcement learning",
        "supervised learning", "unsupervised learning", "neural networks", "cnn", "rnn", "lstm", "transformers", "bert",
        "gpt", "k-means", "decision trees", "random forest", "xgboost", "lightgbm", "bayesian networks", "statistical modeling",
        "data preprocessing", "feature engineering", "dimensionality reduction", "clustering", "regression", "classification",
        "time series analysis", "anomaly detection", "graph neural networks", "recommender systems"
    ]
    
    # Create a pattern to match skills
    pattern = r'\b(' + '|'.join(ai_ml_skills) + r')\b'
    
    # Find all matches in the description
    matches = re.findall(pattern, description.lower())
    
    # Remove duplicates while preserving order
    unique_skills = []
    for skill in matches:
        if skill not in unique_skills:
            unique_skills.append(skill)
    
    return ", ".join(unique_skills) if unique_skills else "Not specified"
# def extract_skills(description):
#     # List of common tech skills to look for
#     common_skills = [
#         "java", "python", "javascript", "react", "node.js", "angular", "vue.js", "html", "css",
#         "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "aws", "azure", "gcp",
#         "docker", "kubernetes", "jenkins", "git", "devops", "ci/cd", "agile", "scrum",
#         "spring", "hibernate", "rest", "soap", "microservices", "c++", "c#", ".net",
#         "machine learning", "ai", "data science", "big data", "hadoop", "spark", "kafka",
#         "php", "ruby", "rails", "django", "flask", "laravel", "swift", "kotlin", "android",
#         "ios", "mobile development", "react native", "flutter", "blockchain", "cybersecurity",
#         "cloud computing", "serverless", "terraform", "ansible", "jira", "confluence"
#     ]
    
#     # Create a pattern to match skills
#     pattern = r'\b(' + '|'.join(common_skills) + r')\b'
    
#     # Find all matches in the description
#     matches = re.findall(pattern, description.lower())
    
#     # Remove duplicates while preserving order
#     unique_skills = []
#     for skill in matches:
#         if skill not in unique_skills:
#             unique_skills.append(skill)
    
#     return ", ".join(unique_skills) if unique_skills else "Not specified"

# Scrape details for each job
print("Scraping job details...")
for j, job_id in enumerate(job_ids):
    try:
        print(f"Scraping job {j+1}/{len(job_ids)}")
        
        # Add a random delay to avoid getting blocked
        time.sleep(random.uniform(1, 3))
        
        # Send the request to LinkedIn
        resp = requests.get(job_url.format(job_id), headers=headers)
        
        # Check if the request was successful
        if resp.status_code != 200:
            print(f"Failed to retrieve job {j+1}. Status code: {resp.status_code}")
            continue
            
        # Parse the HTML content
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Create a new dictionary for this job
        job_info = {}
        
        # Add job ID to the dictionary
        job_info["Job ID"] = job_id
        
        # Extract job title
        try:
            title_elem = soup.find("div", {"class": "top-card-layout__entity-info"})
            if title_elem and title_elem.find("a"):
                job_info["Title"] = title_elem.find("a").text.strip()
            else:
                # Alternative method to find job title
                title_elem = soup.find("h1", {"class": "topcard__title"})
                if title_elem:
                    job_info["Title"] = title_elem.text.strip()
                else:
                    job_info["Title"] = "Not specified"
        except Exception as e:
            job_info["Title"] = "Not specified"
            print(f"Error extracting job title: {e}")
        
        # Extract company name
        try:
            company_elem = soup.find("div", {"class": "top-card-layout__card"})
            if company_elem and company_elem.find("a") and company_elem.find("a").find("img"):
                job_info["Company"] = company_elem.find("a").find("img").get('alt')
            else:
                # Alternative method to find company name
                company_elem = soup.find("a", {"class": "topcard__org-name-link"})
                if company_elem:
                    job_info["Company"] = company_elem.text.strip()
                else:
                    job_info["Company"] = "Not specified"
        except Exception as e:
            job_info["Company"] = "Not specified"
            print(f"Error extracting company name: {e}")
        
        # Extract location
        try:
            location_elem = soup.find("span", {"class": "topcard__flavor--bullet"})
            if location_elem:
                job_info["Location"] = location_elem.text.strip()
            else:
                # Alternative method to find location
                location_elem = soup.find("span", {"class": "job-search-card__location"})
                if location_elem:
                    job_info["Location"] = location_elem.text.strip()
                else:
                    job_info["Location"] = "Not specified"
        except Exception as e:
            job_info["Location"] = "Not specified"
            print(f"Error extracting location: {e}")
        
        # Extract job description
        description = "Not specified"
        try:
            description_elem = soup.find("div", {"class": "description__text"})
            if description_elem:
                description = description_elem.text.strip()
            else:
                # Alternative method to find description
                description_elem = soup.find("div", {"class": "show-more-less-html__markup"})
                if description_elem:
                    description = description_elem.text.strip()
        except Exception as e:
            print(f"Error extracting job description: {e}")
        
        # Extract experience from job description
        job_info["Experience"] = extract_experience(description)
        
        # Extract skills from job description
        job_info["Skills"] = extract_skills(description)
        
        # Add job URL
        job_info["Job URL"] = f"https://www.linkedin.com/jobs/view/{job_id}"
        
        # Add the job info to the list of all jobs
        all_jobs.append(job_info)
        
    except Exception as e:
        print(f"Error scraping job {j+1}: {e}")
        continue

print(f"Successfully scraped {len(all_jobs)} jobs")

# Create a DataFrame from the list of jobs
df = pd.DataFrame(all_jobs)

# Ensure the columns are in the correct order
columns = ["Job ID", "Title", "Company", "Location", "Experience", "Skills", "Job URL"]
df = df[columns]

# Save the DataFrame to a CSV file
filename = f'linkedin_ml_jobs_India.csv'
df.to_csv(filename, index=False, encoding='utf-8')

print(f"Job data saved to {filename}")
print("CSV format includes: Job ID, Title, Company, Location, Experience, Skills, Job URL")