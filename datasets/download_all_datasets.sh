#!/usr/bin/env bash

set -eo pipefail
echo "Downloading all datasets..."

echo "FairFace:"
cd fairface
./download_fairface.py
cd ..

echo "UTKFace:"
cd utkface
./download_utkface.py
cd ..

echo "IAT human data:"
cd iat
./download_age_iat.sh
cd ..

echo "Flickr30k:"
cd utkface
./download_flickr30k.py
cd ..

echo "Done downloading all datasets."
