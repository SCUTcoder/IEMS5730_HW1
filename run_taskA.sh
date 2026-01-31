#!/bin/bash
# Script to run Task A - Find company with max common suppliers

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_path> <output_path>"
    echo "Example: $0 /user/hadoop/medium_relation /user/hadoop/taskA_output"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
JAR_FILE="target/community-detection-1.0-SNAPSHOT.jar"

# Remove output directory if it exists
hadoop fs -rm -r -f ${OUTPUT_PATH}
hadoop fs -rm -r -f ${OUTPUT_PATH}_job1
hadoop fs -rm -r -f ${OUTPUT_PATH}_job2

echo "Running Task A..."
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"

# Run the MapReduce job
hadoop jar ${JAR_FILE} edu.cuhk.iems5730.TaskA ${INPUT_PATH} ${OUTPUT_PATH}

if [ $? -eq 0 ]; then
    echo "Task A completed successfully!"
    echo "View results with: hadoop fs -cat ${OUTPUT_PATH}/part-*"
else
    echo "Task A failed!"
    exit 1
fi
