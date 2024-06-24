#!/bin/bash

source ./do/common.sh


DellIfExists ${WS}/build
ActivateVENV "$venvName"


$pyt slidecoach/main_coach.py
[ $? -ne 0 ] && ExitError


ExitOk