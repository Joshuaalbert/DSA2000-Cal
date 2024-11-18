#!/bin/bash

# Directory containing the log files
LOG_DIR="/tmp/ray/session_*/logs"

# Find and remove backed-up log files matching the pattern '*.1.log', '*.2.log', etc.
find $LOG_DIR -type f -regex '.*/[^/]*\.[0-9]+\.log' -exec rm -f {} \;
