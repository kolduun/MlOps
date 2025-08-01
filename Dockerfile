# app/Dockerfile
FROM python:3.10-slim

WORKDIR /fast_app            

COPY . .    
RUN pip3 install -r ./requirements.txt

EXPOSE 8005

CMD ["python", "./main.py"] 