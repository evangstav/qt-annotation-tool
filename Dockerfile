FROM continuumio/miniconda:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml


COPY / .

CMD ["bash"]
