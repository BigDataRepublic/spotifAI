# Base image
FROM python:3.8-slim-buster

WORKDIR /task

RUN apt-get update
RUN apt-get install -y wget firefox-esr

RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.24.0/geckodriver-v0.24.0-linux64.tar.gz
RUN tar -xvzf geckodriver*

ENV PATH=$PATH:/task

RUN pip install -r requirements.txt

RUN mkdir /task

COPY . /task/

CMD ["/usr/local/bin/python", "top_200_scraper.py"]
