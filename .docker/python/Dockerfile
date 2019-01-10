FROM python:3.7-slim AS build
ENV LANG=C.UTF-8 \
    FLASK_APP=index.py \
    APP_HOME=/var/www/naive-bayes
ENV FLASK_ENV ${FLASK_ENV:-production}
ARG FLASK_ENV

WORKDIR ${APP_HOME}
ADD ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM build
ADD . ${APP_HOME}
EXPOSE 5000
CMD flask run
