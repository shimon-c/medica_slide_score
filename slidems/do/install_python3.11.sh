#!/bin/bash

source ./do/common.sh

libs="python3.11 python3.11-dev python3.11-venv python3.11-tk python3.11-distutils python3-pip"
pyt=python3.11


apt update
[ $? -ne 0 ] && ExitError
add-apt-repository ppa:deadsnakes/ppa -y
[ $? -ne 0 ] && ExitError
apt update
[ $? -ne 0 ] && ExitError
apt install ${libs}
[ $? -ne 0 ] && ExitError
${pyt} --version
[ $? -ne 0 ] && ExitError


ExitOk
