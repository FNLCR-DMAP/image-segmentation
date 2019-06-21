#!/bin/bash

my_dir=$(cd `dirname $0` && pwd)

src_dir=$my_dir/../../../
#TODO: Set imageaug directory or source conda env
if [ -z "$IMGAUG" ]
then
    echo "ERROR: IMGAUG variable must be set" >&2
    exit 1
fi
imgaug_dir=$IMGAUG
export PYTHONPATH=$PYTHONPATH:$src_dir:$imgaug_dir

if [ -z  "$1" ]
then
    echo "need a python script to run" >&2
    echo "example: ./run-test.sh test.py" >&2
    exit
fi
python $1 
