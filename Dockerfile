FROM jupyter/scipy-notebook

COPY apply_normative_models.py ./
COPY models/* models/
COPY docs/* docs/

RUN pip install pcntoolkit==0.20
RUN pip install dash
RUN pip install flask
RUN pip install plotly
