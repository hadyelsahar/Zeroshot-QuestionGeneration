# A NAACL-HLT 2018 ACCEPTED PUBLICATION 
## Zero-Shot Question Generation from Knowledge Graphs for Unseen Predicates and Entity Types
### Authors Hady Elsahar, Christophe Gravier and Frederique Laforest.
[Preprint](./Zeroshot_Question_Generation.pdf)

*The Github Repo consists of the following folders*:

- `baselines` :  containing all the baselines `RTransE`,`IR`,`SELECT`
- `models`: containing code for the encoder-decoder baseline `triples2seq.py` and `our-model`: `tripletext2seq.py`
- `kbembeddings`: code forked and adapted from https://github.com/thunlp/KB2E
- `evaluator`: code for calculating BLEU, METEOR, ROUGE forked and adapted from https://github.com/tylin/coco-caption
- `data`: containing data for training and testing as well as `data.py` python modules that is responsible for creating training/test sets in zeroshot or normal setups.


## Running Model

The file run.py takes some arguments described in the begining of the file and responsible for running training the our-models or the encoder-decoder baseline. 

**Experiment params**:
- `epochs`: number of epochs to train
- `setup`: either zeroshot or normal train/test/valid  splits using the regular splits from SimpleQuestions
- `criteria`: the zeroshot criteria `pred` for unseen pred, `subtype` or `objtype`
- `min`: minimum number of samples for each predicate to keep 
- `datapath`: path of the folder with the preprocessed files
- `kfolds` : number of kfolds 
- `fold`: 0-9 which cross validation split 

**picking model params**:
- `pred`: add textual context for predicates  (flag with no values)
- `subtype` : add textual context for subject types  (flag with no values)
- `objtype` : add textual context for object types  (flag with no values)

**Examples**:

How to run zeroshot experiments for unseen-predicates :


**`Encoder-Decoder`:  No textual contexts or copy actions:**

`python run.py -epochs 10 -min 50 -setup zeroshot -criteria pred -datapath ./data/nocopy -fold 1`

**`Our-Model` the model with textual contexts but no copy actions:**

`python run.py -epochs 10 -min 50 -setup zeroshot -criteria pred  -fold 1 -datapath ./data/nocopy/ -subtype -objtype -pred`

**`Our-Model+Copy` full model with textual contexts and copy actions:**

`python run.py -epochs 10 -min 50 -setup zeroshot -criteria pred -fold 1  datapath ./data/copy/ -subtype -objtype -pred`



# Browsing predicate textual evidence:
The file  `./data/pred_textual_contexts.csv` contains textual contexts collected for the freebase relations.







