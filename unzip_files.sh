#!/bin/bash

for file in ./tmp/*.zip
# for file in ./tmp/evaluation.zip
do
  # UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -d ./data/ "$file"
  UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -d ./data/ "$file"
done