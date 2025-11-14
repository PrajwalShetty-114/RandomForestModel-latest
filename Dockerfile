# --- STAGE 1: The "LFS Fetcher" ---
# We use a base image that has git and git-lfs
# This stage will clone the repo and pull the LFS files
FROM alpine/git:latest AS lfs-fetcher

# Set the GitHub repo URL and branch
ARG REPO_URL="https://github.com/PrajwalShetty-114/RandomForestModel-latest.git"
ARG BRANCH="master"

WORKDIR /repo

# Clone the specific branch
RUN git clone --branch ${BRANCH} ${REPO_URL} .

# Install LFS and pull the large files
RUN git lfs install
RUN git lfs pull

# --- STAGE 2: The "Application" ---
# Now we build the actual Python app
FROM python:3.11-slim

WORKDIR /app

# Install Python requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files (including the LFS-pulled .pkl file)
# from the first stage
COPY --from=lfs-fetcher /repo/data/ /app/data/
COPY --from=lfs-fetcher /repo/main.py .
COPY --from=lfs-fetcher /repo/data/preprocessing.py /app/data/

# Command to run the app
# Render automatically provides the $PORT variable
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT"]
