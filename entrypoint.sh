#!/bin/bash
# this sources oneAPI components and silences the output so we don't
# have to see the wall of text every time we enter the container
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
exec "$@"
