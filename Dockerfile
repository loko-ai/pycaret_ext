FROM python:3.7-slim
RUN apt-get update && apt-get install -y libgomp1
ADD ./requirements.txt /
RUN pip install -r /requirements.txt
ARG GATEWAY
ENV GATEWAY=$GATEWAY
ADD . /plugin
ENV PYTHONPATH=$PYTHONPATH:/plugin
WORKDIR /plugin/services
EXPOSE 8080
CMD python services.py