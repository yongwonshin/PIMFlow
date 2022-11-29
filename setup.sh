#!/bin/bash
if [ ! -d ./mobilenet-v2 ]; then
  tar -xzf ./data/mobilenet-v2.tar.gz -C .
fi
(cd pim && make clean && make)
