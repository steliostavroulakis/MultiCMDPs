FROM python:3

# Install JupyterLab
RUN python -m pip install jupyterlab

# Copy the requirements.txt in a seperate build (https://stackoverflow.com/questions/34398632)
COPY requirements.txt /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Install requirements
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the default JupyterLab port
EXPOSE 8888

# Run JupyterLab when the container launches
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]


# To create a docker container with this dockerfile, run:
# docker build -t myimage .
