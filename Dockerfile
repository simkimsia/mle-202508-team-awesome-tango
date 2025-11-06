# Use official Apache Airflow 2.10.3 image (more stable than 2.11.x)
FROM apache/airflow:2.10.3-python3.11

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark's scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

# Set JAVA_HOME dynamically - create symlink that works for any architecture
RUN JAVA_HOME_DIR=$(dirname $(dirname $(readlink -f $(which java)))) && \
    ln -sf $JAVA_HOME_DIR /usr/lib/jvm/default-java
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Create directories for pipeline data
RUN mkdir -p /opt/airflow/data \
             /opt/airflow/scripts/datamart/bronze \
             /opt/airflow/scripts/datamart/silver \
             /opt/airflow/scripts/datamart/gold \
             /opt/airflow/scripts/datamart/inference \
             /opt/airflow/scripts/datamart/monitoring \
             /opt/airflow/scripts/model_bank \
             /opt/airflow/scripts/docs && \
    chown -R airflow:root /opt/airflow

# Set the working directory
WORKDIR /opt/airflow

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install PyTorch CPU version for cross-platform compatibility (Mac, Windows, Linux)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt