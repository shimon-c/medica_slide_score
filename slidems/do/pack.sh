#!/bin/bash

source ./do/common.sh


DellIfExists ${WS}/build
ActivateVENV "$venvName"


[ "$doClean" = true ] && DellIfExists "${WS_DIST_PATH}"
echo "Creating Runtime distribution package for - $PWD"
$pyt -m pip install --upgrade pip
[ $? -ne 0 ] && ExitError "Line:$LINENO"
$pyt -m pip install --upgrade build setuptools
[ $? -ne 0 ] && ExitError "Line:$LINENO"
$pyt -m build --wheel --outdir=${WS_DIST_PATH}
[ $? -ne 0 ] && ExitError "Line:$LINENO"
echo "whl pkg created at - ${WS_DIST_PATH}"
