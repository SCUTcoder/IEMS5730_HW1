#!/bin/bash
# Upload datasets to HDFS

DATASETS_DIR="./datasets"

echo "Uploading datasets to HDFS..."

# Create HDFS directories
hadoop fs -mkdir -p /user/$(whoami)/supply_chain

# Upload small dataset
if [ -f "${DATASETS_DIR}/small_relation" ]; then
    echo "Uploading small dataset..."
    hadoop fs -put -f ${DATASETS_DIR}/small_relation /user/$(whoami)/supply_chain/
    hadoop fs -put -f ${DATASETS_DIR}/small_label /user/$(whoami)/supply_chain/
fi

# Upload medium dataset
if [ -f "${DATASETS_DIR}/medium_relation" ]; then
    echo "Uploading medium dataset..."
    hadoop fs -put -f ${DATASETS_DIR}/medium_relation /user/$(whoami)/supply_chain/
    hadoop fs -put -f ${DATASETS_DIR}/medium_label /user/$(whoami)/supply_chain/
fi

# Upload large dataset
if [ -f "${DATASETS_DIR}/large_relation" ]; then
    echo "Uploading large dataset..."
    hadoop fs -put -f ${DATASETS_DIR}/large_relation /user/$(whoami)/supply_chain/
    hadoop fs -put -f ${DATASETS_DIR}/large_label /user/$(whoami)/supply_chain/
fi

echo "Upload complete! List files with: hadoop fs -ls /user/$(whoami)/supply_chain/"
