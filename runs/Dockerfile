FROM python:3.10.8

WORKDIR /code
RUN mkdir /gains && mkdir /logs

ADD requirements.txt /code

# Py-spy
RUN pip install -r requirements.txt && pip quartical

COPY ./selfcal.yaml /code

ENTRYPOINT [ "goquartical", "selfcal.yaml" ]