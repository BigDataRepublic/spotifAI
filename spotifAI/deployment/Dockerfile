# specifying base image
FROM python:3.8-slim-buster

COPY . /app

WORKDIR /app

# install all python requirements
RUN pip install -r requirements.txt
RUN pip install -U python-dotenv

EXPOSE 8082

# Command to run the script that launches a flask application
CMD ["python", "publish_playlist.py"]
