#! /bin/bash

wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir glove
mv glove.6B.zip glove
cd glove
unzip glove.6B.zip
rm glove.6B.zip
cd ..
