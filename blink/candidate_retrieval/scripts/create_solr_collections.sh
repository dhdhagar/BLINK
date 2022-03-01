#!/bin/bash




for collection_name in $@
do
    echo "=====Creating collection with name '$collection_name'====="
    sudo su - solr -c "/opt/solr/bin/solr create -c $collection_name -n data_driven_schema_configs"
done
