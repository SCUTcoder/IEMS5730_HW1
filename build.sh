#!/bin/bash
# Build script for MapReduce jobs

echo "Building MapReduce project..."

# Clean and compile
mvn clean package

if [ $? -eq 0 ]; then
    echo "Build successful! JAR file created at target/community-detection-1.0-SNAPSHOT.jar"
else
    echo "Build failed!"
    exit 1
fi
