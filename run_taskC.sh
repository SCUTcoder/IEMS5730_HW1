#!/bin/bash
# Script to run Task C - Count common suppliers by community

if [ $# -lt 3 ]; then
    echo "Usage: $0 <relation_input> <label_input> <output_path>"
    echo "Example: $0 /user/hadoop/medium_relation /user/hadoop/medium_label /user/hadoop/taskC_output"
    exit 1
fi

RELATION_INPUT=$1
LABEL_INPUT=$2
OUTPUT_PATH=$3
JAR_FILE="target/community-detection-1.0-SNAPSHOT.jar"

# Remove output directory if it exists
hadoop fs -rm -r -f ${OUTPUT_PATH}
hadoop fs -rm -r -f ${OUTPUT_PATH}_job1
hadoop fs -rm -r -f ${OUTPUT_PATH}_job2
hadoop fs -rm -r -f ${OUTPUT_PATH}_job3

echo "Running Task C..."
echo "Relation Input: ${RELATION_INPUT}"
echo "Label Input: ${LABEL_INPUT}"
echo "Output: ${OUTPUT_PATH}"

# Run the MapReduce job
hadoop jar ${JAR_FILE} edu.cuhk.iems5730.TaskC ${RELATION_INPUT} ${LABEL_INPUT} ${OUTPUT_PATH}

if [ $? -eq 0 ]; then
    echo "Task C completed successfully!"
    echo "View results with: hadoop fs -cat ${OUTPUT_PATH}/part-*"
else
    echo "Task C failed!"
    exit 1
fi
