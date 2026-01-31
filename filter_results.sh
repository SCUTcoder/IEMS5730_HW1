#!/bin/bash
# Script to filter results based on student ID last 4 digits

if [ $# -lt 2 ]; then
    echo "Usage: $0 <student_id_last_4_digits> <result_file>"
    echo "Example: $0 4321 taskA_output/part-r-00000"
    echo ""
    echo "This will filter results for companies with IDs: 4321, 14321, 24321, etc."
    exit 1
fi

LAST_4_DIGITS=$1
RESULT_FILE=$2

echo "Filtering results for companies ending with ${LAST_4_DIGITS}..."

# Download the result file from HDFS if it's an HDFS path
if [[ $RESULT_FILE == hdfs://* ]] || [[ $RESULT_FILE == /user/* ]]; then
    TEMP_FILE=$(mktemp)
    hadoop fs -cat ${RESULT_FILE} > ${TEMP_FILE}
    RESULT_FILE=${TEMP_FILE}
fi

# Filter lines where company ID ends with the specified digits
grep -E "^[0-9]*${LAST_4_DIGITS}:" ${RESULT_FILE}

# Clean up temp file if created
if [ ! -z "${TEMP_FILE}" ]; then
    rm -f ${TEMP_FILE}
fi
