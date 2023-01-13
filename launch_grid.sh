#!/bin/bash

# Uncomment this to launch a jupyter-lab environment in the browser
#docker run -p 8888:8888 -e DISPLAY=localhost:0 -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOCKEN=docker multicmdp

# Uncomment this to launch a normal docker container
docker run --entrypoint /bin/bash -v $(pwd):/app -p 6000:6000 -e DISPLAY=localhost:0 --rm -i -t multicmdp