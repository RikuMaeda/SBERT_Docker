FROM python:3
USER root

WORKDIR /app

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN python -m pip install jupyterlab
RUN python -m pip install transformers
RUN python -m pip install torch torchvision
RUN python -m pip install fugashi
RUN python -m pip install ipadic
RUN python -m pip install numpy
RUN python -m pip install scikit-learn
RUN python -m pip install  pandas


#CMD ["python", "SBERT.py"]