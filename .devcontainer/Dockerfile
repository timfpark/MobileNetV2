FROM tensorflow/tensorflow:latest-gpu-jupyter

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends python3-opencv ffmpeg libsm6 libxext6

RUN pip install --upgrade pip
RUN pip install opencv-python tensorflow keras pandas pillow scipy
