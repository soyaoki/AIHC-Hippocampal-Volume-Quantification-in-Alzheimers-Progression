#!/bin/bash

apt-get install dcmtk -y

curl -X POST http://localhost:8042/tools/execute-script --data-binary @route_dicoms.lua -v
sudo storescp 106 -v -aet HIPPOAI -od  /home/workspace/out/data --sort-on-study-uid st