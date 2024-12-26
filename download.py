import requests
import subprocess
import os
import shutil
import pickle
from datetime import datetime

# GitHub API URL
API_URL = "https://api.github.com/search/repositories"

# GitHub API Token
GITHUB_TOKEN = "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"

# Headers for API request
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Parameters for the search query
params = {
    "q": "language:python created:>=2024-01-01 stars:>=50",
    "sort": "updated",
    "order": "desc",
    "per_page": 100,
    "page": 1
}

# Directory to save the projects
SAVE_DIR = "github_projects"
os.makedirs(SAVE_DIR, exist_ok=True)

# File to save progress
PROGRESS_FILE = "download_progress.pkl"

# Function to clone a repository
def clone_repo(repo_url, repo_name):
    repo_dir = os.path.join(SAVE_DIR, repo_name)
    if not os.path.exists(repo_dir):
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
            print(f"Successfully cloned {repo_name}")
            # Remove non .py files
            remove_non_py_files(repo_dir)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo_name}: {e}")

# Function to remove non .py files from a directory
def remove_non_py_files(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if not file.endswith(".py"):
                file_path = os.path.join(root, file)
                if os.path.islink(file_path):
                    os.unlink(file_path)
                else:
                    os.remove(file_path)
                print(f"Removed non .py file: {file_path}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if os.path.islink(dir_path):
                os.unlink(dir_path)
            elif not any(fname.endswith(".py") for fname in os.listdir(dir_path)):
                shutil.rmtree(dir_path)
                print(f"Removed empty directory: {dir_path}")

# Function to save progress
def save_progress(target_repo, downloaded_repos):
    with open(PROGRESS_FILE, "wb") as f:
        pickle.dump((target_repo, downloaded_repos), f)

# Function to load progress
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "rb") as f:
            return pickle.load(f)
    return [], 0

# Function to download repositories
def download_repositories(max_repos=1000):
    target_repo, downloaded_repos = load_progress()
    downloaded_repos = 0
    
    while len(target_repo) < max_repos:
        response = requests.get(API_URL, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(len(target_repo))
            print(response)
            exit()
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            print("No more repositories found.")
            exit()
        
        target_repo.extend(items)
        params["page"] += 1
    
    try:
        while downloaded_repos < max_repos:

            for item in target_repo:
                repo_url = item["clone_url"]
                repo_name = item["name"]
                clone_repo(repo_url, repo_name)
                downloaded_repos += 1
                print(f"Downloaded {downloaded_repos}/{max_repos}: {repo_name}")

                if downloaded_repos >= max_repos:
                    break

            # Move to the next page
            params["page"] += 1

    except KeyboardInterrupt:
        print("\nDownload interrupted. Saving progress...")
        save_progress(target_repo, downloaded_repos)
        print("Progress saved. You can resume the download next time.")

# Run the script
download_repositories(max_repos=1000)
