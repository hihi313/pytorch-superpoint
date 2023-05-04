#!/bin/bash

$VENV_NAME=sp_venv

set -e

conda init bash
# conda activate $VENV_NAME

exec "$@"