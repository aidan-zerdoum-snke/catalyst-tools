ARG CONTAINER_REGISTRY="ghcr.io/snkeos"

FROM ${CONTAINER_REGISTRY}/ml-base:ubuntu22.04-py3.12.9-cuda12.4.1-cudnn9.1.0-runtime-uv-latest
# or check https://github.com/snkeos/ana-containers/tree/stable/images/ml-base for more images

COPY --link ./pyproject.toml /usr/src/pyproject.toml

RUN uv pip install --no-cache -r /usr/src/pyproject.toml && \
    chown -R ${USERNAME}:${USERNAME} ${VENV_PATH}

CMD ["/bin/bash"]
