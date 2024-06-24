#!/bin/bash


doClean=false
venvName="venv"
pyt=python3.11
PIPInstallType=development


ThisScriptDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WS=`dirname $ThisScriptDir`
WS_NAME=`basename $WS`
WS_PATH=`dirname $WS`
PROJECT_WS_PATH=`dirname $WS_PATH`
PKG_SRC_NAME=$(sed -n '/^name \?= \?\"\w/p' $WS/pyproject.toml | grep -oP "^name ?= ?\"\K\w+")
[ $? -ne 0 ] && ExitError "Line:$LINENO Error reading pkg name from pyproject.toml"
WS_BUILD_PATH=${WS}/Build
WS_DIST_PATH=${WS}/output/dist_pkg



#---------------------------------------
function ExitError {
    echo " CWD = `pwd`"
    date
    echo "ERROR: $0 - $1"
    echo "Live Long And Prosper"
    echo
    exit 1
}

#---------------------------------------
function ExitOk {
    echo " CWD = `pwd`"
    date
    echo "Live Long And Prosper"
    echo
    exit 0
}

#---------------------------------------
function DellIfExists {
    if [ -e $1 ]; then
        echo " deleting - $1"
        rm -rf $1
        [ $? -ne 0 ] && ExitError "Line:$LINENO Error deleting - $1"
    else
        echo " $1 not exist"
    fi
}

#---------------------------------------
function CreateIfNotExists {
    if [ -e $1 ]; then
        echo " $1 exist"
    else
        echo " mkdir -p $1"
        mkdir -p $1
        [ $? -ne 0 ] && ExitError "Line:$LINENO Error creating directory - $1"
    fi
}


#---------------------------------------
function ActivateVENV {
    [ "$doClean" = true ] && DellIfExists "${WS_BUILD_PATH}/$1"
    CreateIfNotExists ${WS_BUILD_PATH}
    cd ${WS_BUILD_PATH}
    echo " Creating venv   - ${WS_BUILD_PATH}/$1"
    $pyt -m venv $1
    [ $? -ne 0 ] && ExitError "Line:$LINENO - at ActivateVENV"
    echo " Activating venv - ${WS_BUILD_PATH}/$1"
    source $1/bin/activate
    [ $? -ne 0 ] && ExitError "Line:$LINENO Error - source $1/bin/activate"

    pyt=${WS_BUILD_PATH}/${venvName}/bin/$pyt
    $pyt -c "import sys; print(f' Using Python at - {sys.prefix}')"
    [ $? -ne 0 ] && ExitError "Line:$LINENO - at ActivateVENV"
    cd $WS
}



InARGV=( "$@" )
InARGC=${#@}
for (( n=0 ; n < $InARGC ; n++ )) ; do
    case ${InARGV[$n]} in
        --clean)
            doClean=true
        ;;
        --project_ws)
            n=$((n+1))
            PROJECT_WS_PATH=${InARGV[$n]}
            WS_BUILD_PATH=${PROJECT_WS_PATH}/Build
            WS_DIST_PATH=${PROJECT_WS_PATH}/output/dist_pkg
        ;;
        --install)
            n=$((n+1))
            PIPInstallType=${InARGV[$n]}
        ;;
        --runtime)
            venvName=venv_runtime_${PKG_SRC_NAME}
            runTime=true
        ;;
    esac
done

echo " Package Name = ${PKG_SRC_NAME}"
echo " WS -      $WS"
echo " WS_NAME - $WS_NAME"
echo " WS_PATH - $WS_PATH"
echo " WS_DIST_PATH - ${WS_DIST_PATH}"
echo " PROJECT_WS_PATH - ${PROJECT_WS_PATH}"