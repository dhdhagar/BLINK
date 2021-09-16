#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


for collection_name in $@
do
    echo "=====Creating collection with name '$collection_name'====="
    sudo su - solr -c "/opt/solr/bin/solr create -c $collection_name -n data_driven_schema_configs"
done
