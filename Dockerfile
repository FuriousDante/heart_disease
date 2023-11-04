FROM python:3.11.1

WORKDIR /heartdisdir
COPY . /heartdisdir
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run server.py