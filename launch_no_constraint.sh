#!/bin/bash

# Define the dataset variable
DATASET="dtd"

# Update configuration file with the dataset variable
CONFIG_PATH="./configs/mask/mask_dtd.yaml"
CONFIG_PATH_2="./configs/mask/mask_dtd_binary.yaml"
#CONFIG_PATH_2="./configs/mask/mask_mscoco_binary.yaml"


n=0

#for n in {1..10}
#do
yq eval ".MODEL.N_LOWEST = ${n}" -i ${CONFIG_PATH_2}
#yq eval ".OUTPUT = \"output_files/${n}_HEAD\"" -i ${CONFIG_PATH_2}
for i in {0..49} #loop trhough the task number
do
    #python main_one_task.py --cfg ${CONFIG_PATH} ${i}
    python main_one_task.py --cfg ${CONFIG_PATH_2} ${i}
done
