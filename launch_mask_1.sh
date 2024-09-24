#!/bin/bash

# Define the dataset variable
DATASET="dtd"

# Update configuration file with the dataset variable
CONFIG_PATH="./configs/mask/mask_dtd.yaml"



# Run the Python script for each configuration
for i in {0..49} #loop trhough the tak number
do
    python main_one_task.py --cfg ${CONFIG_PATH} ${i}
done

