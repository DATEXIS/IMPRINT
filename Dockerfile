FROM python:3.10.14-bookworm

RUN apt-get update
RUN apt-get install -y htop tmux

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire project
COPY . /app/
WORKDIR /app

# Install the package in development mode
RUN pip install -e .

# The PYTHONPATH environment variable is not needed since we're installing as a package
# ENV PYTHONPATH=/app

