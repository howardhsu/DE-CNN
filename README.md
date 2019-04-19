# Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction
Code for our ACL 2018 paper "[Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction](http://www.aclweb.org/anthology/P18-2094)"

## Problem to Solve

Label "The retina display is great ." as "O B I O O O" so to extract "retina display" as an aspect.
Check this [article](https://howardhsu.github.io/article/absa/) for aspect-based sentiment analysis or [this](https://howardhsu.github.io/article/tdrl/) for domain representation learning.

## Environment

All code are tested under python 3.6.2 + pytorch 0.2.0_4

## Steps to Run Code

Step 1: Download general embeddings (GloVe: http://nlp.stanford.edu/data/glove.840B.300d.zip ), save it in data/embedding/gen.vec 

Step 2: Download Domain Embeddings (You can find the link under this paper's title in https://www.cs.uic.edu/~hxu/ ), save them in data/embedding

Step 3:
Download and install fastText (https://github.com/facebookresearch/fastText) to fastText/

Step 4: 
Download official datasets to data/official_data/

Download official evaluation scripts to script/

We assume the following file names.

SemEval 2014 Laptop (http://alt.qcri.org/semeval2014/task4/):

data/official_data/Laptops_Test_Data_PhaseA.xml

data/official_data/Laptops_Test_Gold.xml

script/eval.jar

SemEval 2016 Restaurant (http://alt.qcri.org/semeval2016/task5/)

data/official_data/EN_REST_SB1_TEST.xml.A

data/official_data/EN_REST_SB1_TEST.xml.gold

script/A.jar

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

## Citation

If you find our code useful, please cite our paper.
```
@InProceedings{xu_acl2018,
  author    = {Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S.},
  title     = {Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  publisher = {Association for Computational Linguistics},
  year      = {2018}
}
```
