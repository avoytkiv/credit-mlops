FROM python:3.11
ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONPATH=/app
COPY . .
RUN pip install -r requirements.txt 
RUN ls -la  # List files in the working directory
# list files in models directory
RUN ls -la models
CMD uvicorn app.main:app --host=0.0.0.0 --port $PORT --reload