#!/bin/bash

python /home/workspace/src/inference_dcm.py /data/TestVolumes/Study3

bash /home/workspace/src/deploy_scripts/send_volume.sh

bash /home/workspace/src/deploy_scripts/send_result.sh 