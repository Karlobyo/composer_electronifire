FROM python:3.10.6-buster
EXPOSE 8501
COPY . .
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
# CMD uvicorn ./app:app --host 0.0.0.0 --port 8000
#ENTRYPOINT ["composer-electronifier", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# RUN pip uninstall protobuf
# RUN pip install protobuf==3.19.6
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
