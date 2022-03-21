#! /bin/bash

echo Create prompt directory:
mkdir prompts

echo "Downloading human result from 2019 Age IAT..."
wget -O prompts/age_iat_2019.zip https://osf.io/wy65p/download
unzip prompts/age_iat_2019.zip -d human_data

echo "Downloading human result from 2021 Gender Science IAT..." # TODO: confirm year
wget -O prompts/gender_science_iat_2021.zip https://osf.io/uha9k/download
unzip prompts/gender_science_iat_2021.zip -d prompts

echo "Downloading human result from 2021 Race IAT..."
wget -O prompts/race_iat_2021.zip https://osf.io/3k278/download
unzip prompts/race_iat_2021.zip -d prompts

echo "Downloading human result from Pairwise Adjectives IAT..."
wget -O prompts/pairwise_adjectives_iat_2021.zip https://osf.io/wy65p/download # TODO: find correct link, confirm year
unzip prompts/pairwise_adjectives_iat_2021.zip -d prompts