FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TZ=Asia/Taipei

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_DPDS=apt_packages.txt
ARG PY_DPDS=requirements.txt

WORKDIR /tmp

USER root

RUN apt update

# install apt dependencies
COPY ./${APT_PKGS} ./
RUN xargs apt install \
    --yes \
    --no-install-recommends \
    < ${APT_DPDS}

# install python dependencies
COPY ./${PY_DPDS} ./
RUN python3 -m pip install --upgrade pip setuptools \
    && python3 -m pip install \
    --no-cache-dir \
    --find-links https://download.pytorch.org/whl/cu111/torch_stable.html \
    --requirement ${PY_DPDS}

# Clean up
RUN apt autoremove --purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# setup entrypoint
# COPY ./eentrypoint.sh /
# ENTRYPOINT ["/entrypoint.sh"]

# USER user

WORKDIR /app

CMD ["bash"]

# TODO: add python2 venv? (for eval tool: https://github.com/uzh-rpg/rpg_trajectory_evaluation)
# RUN apt install -y -no-recom python3-venv=3.8.2-0ubuntu2
# ADD https://bootstrap.pypa.io/pip/2.7/get-pip.py
# RUN python3 -m venv myenv
# RUN source myenv/bin/activate