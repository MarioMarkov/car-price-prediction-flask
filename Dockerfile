# start by pulling the python image
FROM python:3.9

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
# RUN apk add curl
# RUN apk add gcc
# RUN curl -proto '=https' -tlsv1.2 -y -sSf https://sh.rustup.rs | sh
# RUN source $HOME/.cargo/env
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]