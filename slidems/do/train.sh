#!/bin/bash

source ./do/common.sh


DellIfExists ${WS}/build
ActivateVENV "$venvName"


$pyt slidems/train/train.py
[ $? -ne 0 ] && ExitError


ExitOk