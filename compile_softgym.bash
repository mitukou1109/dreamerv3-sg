#!/usr/bin/bash

WORKDIR=$(pwd)

cd ${PYFLEXROOT}/bindings
rm -rf build
mkdir build
cd build

cmake -DPYBIND11_PYTHON_VERSION=3.8 ..
make -j

cd ${WORKDIR}