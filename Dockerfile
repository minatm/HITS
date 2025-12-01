# ---------------------------------------------------------
# Base Python image
# ---------------------------------------------------------
FROM python:3.10-slim

# ---------------------------------------------------------
# Install system dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install Gurobi
# ---------------------------------------------------------
WORKDIR /opt

# Download Gurobi 10.0.3
RUN wget https://packages.gurobi.com/10.0/gurobi10.0.3_linux64.tar.gz && \
    tar -xvf gurobi10.0.3_linux64.tar.gz && \
    rm gurobi10.0.3_linux64.tar.gz

# Set GUROBI environment variables
ENV GUROBI_HOME="/opt/gurobi10.0.3/linux64"
ENV PATH="${GUROBI_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"

# ⭐ THE MISSING LINE — enables WLS cloud licensing ⭐
ENV GRB_USEMLIC=1

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------------------------------------------------------
# Copy your Streamlit app to the container
# ---------------------------------------------------------
COPY . /app

# ---------------------------------------------------------
# Expose Streamlit default port
# ---------------------------------------------------------
EXPOSE 8501

# ---------------------------------------------------------
# Start the Streamlit app
# ---------------------------------------------------------
CMD ["streamlit", "run", "hits-schedule.py", "--server.port=8501", "--server.address=0.0.0.0"]
