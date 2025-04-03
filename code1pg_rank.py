# import pandas as pd
# import networkx as nx

# # Load the job data from CSV
# df = pd.read_csv("linkedin_ml_jobs_India.csv")

# # Create a graph
# G = nx.DiGraph()

# # Add job nodes
# for index, row in df.iterrows():
#     job_id = row["Job ID"]
#     G.add_node(job_id, title=row["Title"], company=row["Company"], skills=row["Skills"])

# # Add edges based on similarity (title, skills, company)
# for i, job1 in df.iterrows():
#     for j, job2 in df.iterrows():
#         if i != j:
#             weight = 0
            
#             # Higher weight for same job title
#             if job1["Title"] == job2["Title"]:
#                 weight += 2  
            
#             # Moderate weight for same company
#             if job1["Company"] == job2["Company"]:
#                 weight += 1  
            
#             # Higher weight for common skills
#             common_skills = set(job1["Skills"].split(", ")) & set(job2["Skills"].split(", "))
#             weight += len(common_skills)  

#             # Add edge if weight > 0
#             if weight > 0:
#                 G.add_edge(job1["Job ID"], job2["Job ID"], weight=weight)

# # Apply PageRank
# pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")

# # Sort jobs based on PageRank score
# sorted_jobs = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

# # Convert sorted results into DataFrame
# ranked_jobs = pd.DataFrame(sorted_jobs, columns=["Job ID", "PageRank Score"])
# final_df = df.merge(ranked_jobs, on="Job ID").sort_values(by="PageRank Score", ascending=False)

# # Save to CSV
# final_df.to_csv("ranked_jobs1.csv", index=False)

# print("Job ranking completed! Check ranked_jobs.csv")
# # def recommend_jobs(skill, location, top_n=5):
# #     filtered_jobs = final_df[(final_df["Skills"].str.contains(skill, case=False)) &
# #                              (final_df["Location"].str.contains(location, case=False))]
    
# #     recommended = filtered_jobs.sort_values(by="PageRank Score", ascending=False).head(top_n)
# #     return recommended[["Title", "Company", "Location", "Skills", "PageRank Score"]]

# # # Example: Recommend top 5 Python jobs in Bangalore
# # print(recommend_jobs("react", "Bengaluru"))


###########111
# import pandas as pd
# import networkx as nx

# # Load job data
# df = pd.read_csv("linkedin_ml_jobs_India.csv")

# # Create graph
# G = nx.DiGraph()

# # Add job nodes
# for index, row in df.iterrows():
#     job_id = row["Job ID"]
#     G.add_node(job_id, title=row["Title"], company=row["Company"], skills=row["Skills"], experience=row["Experience"])

# # Add edges based on similarity
# for i, job1 in df.iterrows():
#     for j, job2 in df.iterrows():
#         if i != j:
#             weight = 0
            
#             # Higher weight for same job title
#             if job1["Title"] == job2["Title"]:
#                 weight += 2  
            
#             # Moderate weight for same company
#             if job1["Company"] == job2["Company"]:
#                 weight += 1  

#             # Increase weight for common skills
#             skills1 = set(str(job1["Skills"]).split(", "))
#             skills2 = set(str(job2["Skills"]).split(", "))
#             common_skills = skills1 & skills2
#             weight += len(common_skills) * 3  # Increased weight

#             # Experience similarity (Optional: Define your own similarity function)
#             exp1 = str(job1["Experience"]).split()[0]  # Extract first number
#             exp2 = str(job2["Experience"]).split()[0]  
#             if exp1.isdigit() and exp2.isdigit():
#                 if abs(int(exp1) - int(exp2)) <= 2:  # If experience differs by max 2 years
#                     weight += 2  

#             # Add edge if weight > 0
#             if weight > 0:
#                 G.add_edge(job1["Job ID"], job2["Job ID"], weight=weight)

# # Apply PageRank with weight
# pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")

# # Sort jobs based on PageRank score
# sorted_jobs = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

# # Convert sorted results into DataFrame
# ranked_jobs = pd.DataFrame(sorted_jobs, columns=["Job ID", "PageRank Score"])
# final_df = df.merge(ranked_jobs, on="Job ID").sort_values(by="PageRank Score", ascending=False)

# # Save to CSV
# final_df.to_csv("ranked_jobs12.csv", index=False)

# print("Job ranking completed! Check ranked_jobs12.csv")



#######12222



import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

# Load dataset
file_path = "linkedin_ml_jobs_India.csv"  # Update this with the correct CSV path
df = pd.read_csv(file_path)

# Build job similarity graph (Nodes: Jobs, Edges: Similar Jobs)
G = nx.DiGraph()

# Add nodes (Job IDs)
for index, row in df.iterrows():
    G.add_node(row["Job ID"], title=row["Title"], company=row["Company"])

# Function to calculate skill similarity (Jaccard Index)
def skill_similarity(skills1, skills2):
    set1, set2 = set(skills1.split(", ")), set(skills2.split(", "))
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

# Create edges between jobs with similar skills (Threshold: 0.3)
for (idx1, job1), (idx2, job2) in combinations(df.iterrows(), 2):
    sim_score = skill_similarity(job1["Skills"], job2["Skills"])
    if sim_score > 0.3:  # Only add links if similarity > 30%
        G.add_edge(job1["Job ID"], job2["Job ID"], weight=sim_score)
        G.add_edge(job2["Job ID"], job1["Job ID"], weight=sim_score)

# Compute PageRank with adjusted damping factor
pagerank_scores = nx.pagerank(G, alpha=0.85)

# Normalize scores (Scaling 0-1 for better readability)
scaler = MinMaxScaler()
df["PageRank Score"] = df["Job ID"].map(pagerank_scores)
df["PageRank Score"] = scaler.fit_transform(df[["PageRank Score"]])

# Save updated dataset
df.to_csv("updated_jobs1.csv", index=False)

print("Updated PageRank scores saved to 'updated_jobs1.csv'.")
