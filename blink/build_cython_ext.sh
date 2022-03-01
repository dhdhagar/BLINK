#!/bin/bash

set -exu

pushd joint/joint_eval
rm special_partition.c*
rm special_partition.html
cythonize -a -i special_partition.pyx
popd
