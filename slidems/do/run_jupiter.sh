#!/bin/bash

source ./do/common.sh


DellIfExists ${WS}/build
ActivateVENV "$venvName"


jupyter lab
[ $? -ne 0 ] && ExitError

ExitOk