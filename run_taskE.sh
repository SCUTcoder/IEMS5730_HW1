#!/bin/bash
# Script to run Task E - TOP-K similar companies for large dataset

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_path> <output_path> [K]"
    echo "Example: $0 /user/hadoop/large_relation /user/hadoop/taskE_output 4"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
K=${3:-4}  # Default K=4
JAR_FILE="target/community-detection-1.0-SNAPSHOT.jar"

# Remove output directory if it exists
hadoop fs -rm -r -f ${OUTPUT_PATH}
hadoop fs -rm -r -f ${OUTPUT_PATH}_job1
hadoop fs -rm -r -f ${OUTPUT_PATH}_job2

echo "Running Task E with K=${K}..."
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Note: This may take a long time for large datasets..."

# Run the MapReduce job with compression enabled
hadoop jar ${JAR_FILE} edu.cuhk.iems5730.TaskE ${INPUT_PATH} ${OUTPUT_PATH} ${K}

if [ $? -eq 0 ]; then
    echo "Task E completed successfully!"
    echo "View results with: hadoop fs -cat ${OUTPUT_PATH}/part-*"
else
    echo "Task E failed!"
    exit 1
fi
