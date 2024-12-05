# we are using 3.9.19 due to fixed versions of the packages in the requirements.txt
FROM python:3.9.19 

# Set the working directory
WORKDIR /workspace

COPY requirements.txt .

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the SDK requirements
RUN pip install -r requirements.txt
    

# Set the default command to bash (optional)
CMD ["bash"]
