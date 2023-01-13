FROM python:3

# Install JupyterLab
RUN python -m pip install jupyterlab

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the default JupyterLab port
EXPOSE 8888

# Run JupyterLab when the container launches
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]


# To create a docker container with this dockerfile, run:
# docker build -t myimage .