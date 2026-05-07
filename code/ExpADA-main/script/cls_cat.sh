#!/bin/bash

# Usage: 
#   cat_rmse.sh <log_file>
# Function: 
#   This script reads a log file and prints the lines containing the rmse values in ascending order


log=$1

# grep to find lines with "rmse", cut to extract the value (assuming it's the second field), and sort numerically
rmse_values=$(cat $log | grep "rmse" | cut -d':' -f4 | cut -d' ' -f2 | sort -n)
echo $rmse_values 
# For each sorted rmse value, find its line number in the original file
# for value in $rmse_values
# do
#     grep -n "$value" $log
# done