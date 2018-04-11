#!/usr/bin/env bash

# downloading pre-trained entity TransE embeddings
wget  https://www.dropbox.com/s/9mrjyjwks6ewqcx/ent_embeddings.zip
mv ent_embeddings.zip ./checkpoints/transe/
cd ./checkpoints/transe/
unzip ent_embeddings.zip
rm ent_embeddings.zip
cd ../../







