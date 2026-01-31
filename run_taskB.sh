#!/bin/bash
# Script to run Task B - Find TOP-K most similar companies

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_path> <output_path> [K]"
    echo "Example: $0 /user/hadoop/medium_relation /user/hadoop/taskB_output 3"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
K=${3:-3}  # Default K=3
JAR_FILE="target/community-detection-1.0-SNAPSHOT.jar"

# Remove output directory if it exists
hadoop fs -rm -r -f ${OUTPUT_PATH}
hadoop fs -rm -r -f ${OUTPUT_PATH}_job1
hadoop fs -rm -r -f ${OUTPUT_PATH}_job2

echo "Running Task B with K=${K}..."
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"

# Run the MapReduce job
hadoop jar ${JAR_FILE} edu.cuhk.iems5730.TaskB ${INPUT_PATH} ${OUTPUT_PATH} ${K}

if [ $? -eq 0 ]; then
    echo "Task B completed successfully!"
    echo "View results with: hadoop fs -cat ${OUTPUT_PATH}/part-*"
else
    echo "Task B failed!"
    exit 1
fi
