FROM python:3.8-slim
# FROM python:3

# set a directory for the app
WORKDIR /app

# copy all the files to the container
COPY . /app

RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python-dev

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# run the command
CMD ["python", "./app_shingles.py"]
