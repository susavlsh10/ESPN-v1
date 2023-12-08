#!/bin/bash

# Output file name
OUTPUT_FILE="output.txt"

# Loop to run the test executable multiple times
NUM_RUNS=20

output_text="Using $NUM_RUNS queries"
echo "$output_text" >> "$OUTPUT_FILE"

for ((i=1; i<=NUM_RUNS; i++)); do
    /home/grads/s/sls7161/nvme/gds/tools/gdsio -f /home/grads/s/sls7161/nvme/float_save/float_268435456_a.dat -n 0 -m 0 -s 4M -d 0 -x 6 -I 2 -w 128 -i 4k -T 1 >> "$OUTPUT_FILE" &
done


# Wait for all background processes to finish
wait
echo -e "\n" >> "$OUTPUT_FILE"
