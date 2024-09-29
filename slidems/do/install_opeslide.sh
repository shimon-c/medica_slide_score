#!/bin/bash

pyt=python3.11
opeslide_libs="openslide-tools python3-openslide"


#---------------------------------------
function ExitError {
    echo "CWD = `pwd`"
    date
    echo "ERROR: $0 - $1"
    echo "Live Long And Prosper"
    echo
    exit 1
}

#---------------------------------------
function ExitOk {
    echo "CWD = `pwd`"
    date
    echo "Live Long And Prosper"
    echo
    exit 0
}


apt update
[ $? -ne 0 ] && ExitError
add-apt-repository ppa:openslide/openslide -y
[ $? -ne 0 ] && ExitError
apt update
[ $? -ne 0 ] && ExitError
apt install ${opeslide_libs}
[ $? -ne 0 ] && ExitError


ExitOk
