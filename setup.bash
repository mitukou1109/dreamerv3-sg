#!/usr/bin/bash

CONDA_ENV_NAME=dreamerv3-sg

conda activate ${CONDA_ENV_NAME}

WORKDIR=$(pwd)
cd ~/anaconda3/envs/${CONDA_ENV_NAME}/lib/python3.8/site-packages/tensorrt_libs
if [[ ! -e libnvinfer.so.7 ]] && [[ -e libnvinfer.so.8 ]]; then
  ln -s libnvinfer.so.8 libnvinfer.so.7
fi
if [[ ! -e libnvinfer_plugin.so.7 ]] && [[ -e libnvinfer_plugin.so.8 ]]; then
  ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7
fi
cd ${WORKDIR}

export DREAMERV3ROOT=~/sakamoto/dreamerv3
export SOFTGYMROOT=~/sakamoto/softgym
export PYFLEXROOT=${SOFTGYMROOT}/PyFlex

export PYTHONPATH=${DREAMERV3ROOT}:${SOFTGYMROOT}:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:~/anaconda3/envs/${CONDA_ENV_NAME}/lib/python3.8/site-packages/tensorrt_libs:~/anaconda3/envs/${CONDA_ENV_NAME}/lib:$LD_LIBRARY_PATH