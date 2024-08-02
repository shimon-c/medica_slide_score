#!/bin/bash


source ./do/common.sh

# ======= Main ========
ActivateVENV "$venvName"

echo " $pyt -m pip install --upgrade pip"
$pyt -m pip install --upgrade pip
[ $? -ne 0 ] && ExitError "Line:$LINENO Error at pip upgrade"

echo " $pyt -m pip install -e $WS[$PIPInstallType] --find-links $WS_PATH"
$pyt -m pip install -e $WS[development] --find-links $WS_PATH
[ $? -ne 0 ] && ExitError "Line:$LINENO Error at pip install"

ExitOk