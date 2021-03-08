# set base image (host OS)
FROM pytorch/pytorch:latest
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY src/ .
EXPOSE 5000
#RUN FLASK_ENV=development FLASK_APP=app.py flask run
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app.py"]
CMD ["python","./app.py"]
