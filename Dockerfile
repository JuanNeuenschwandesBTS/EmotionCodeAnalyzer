FROM python:3.11-slim

ENV APP_DIR /usr/src/app


LABEL author="John, Juan, Ciaran"
LABEL description="Deployment of Emotion Code Analyzer App"

WORKDIR ${APP_DIR}

COPY requirements.txt ${APP_DIR}/

RUN pip install -r requirements.txt

COPY App/ ${APP_DIR}/

COPY Data/Models ${APP_DIR}/

COPY Makefile ${APP_DIR}/

CMD ["make", "all"]