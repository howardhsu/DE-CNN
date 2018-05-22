# Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction
Code for paper "Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction"


All code are tested under python 3.6.2 + pytorch 0.2.0_4
Step 1: Download general embeddings (GloVe: http://nlp.stanford.edu/data/glove.840B.300d.zip ), save it in data/embedding/gen.vec 

Step 2: Download Domain Embeddings (You can find the link under this paper's title in https://www.cs.uic.edu/~hxu/ ), save them in data/embedding

Step 3:
Download and install fastText (https://github.com/facebookresearch/fastText) to fastText/

Step 4: 
Download official datasets to data/official_data/
We assume the following file names.
SemEval 2014 Laptop (http://alt.qcri.org/semeval2014/task4/):
Laptops_Test_Data_PhaseA.xml
Laptops_Test_Gold.xml

SemEval 2016 Restaurant (http://alt.qcri.org/semeval2016/task5/)
EN_REST_SB1_TEST.xml.A
EN_REST_SB1_TEST.xml.gold


Step 5: Run prep_embedding.py to build numpy files for general embeddings and domain embeddings.
```
python script/prep_embedding.py
```

Step 6: Fill in out-of-vocabulary (OOV) embedding
```
./fastText/fasttext print-word-vectors data/embedding/laptop_emb.vec.bin < data/prep_data/laptop_emb.vec.oov.txt > data/prep_data/laptop_oov.vec

./fastText/fasttext print-word-vectors data/embedding/restaurant_emb.vec.bin < data/prep_data/restaurant_emb.vec.oov.txt > data/prep_data/restaurant_oov.vec

python script/prep_oov.py
```

Step 7: Train the laptop model
```
python script/train.py
```
Train the restaurant model
```
python script/train.py --domain restaurant 
```

Step 8: Evaluate Laptop dataset
```
python script/evaluation.py
```
Evaluate Restaurant dataset
```
python script/evaluation.py --domain restaurant 
```

Optional: 
0. install AllenNLP to if you want to use the CRF layer. (https://github.com/allenai/allennlp)
