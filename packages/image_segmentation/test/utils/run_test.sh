#!/bin/bash

my_dir=$(cd `dirname $0` && pwd)

src_dir=$my_dir/../../../
imgaug_dir=/data/HiTIF/data/dl_segmentation_input/utils/imgaug
export PYTHONPATH=$PYTHONPATH:$src_dir:$imgaug_dir

if [ -z  "$1" ]
then
    echo "need a python script to run" >&2
    echo "example: ./run-test.sh test.py" >&2
    exit
fi
python $1 
