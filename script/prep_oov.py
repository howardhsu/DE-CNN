import argparse
import numpy as np
import json
import subprocess

def fill_np_embedding(emb_file, word_idx_fn, oov_fn):
    with open(word_idx_fn) as f:
        word_idx=json.load(f)
    embedding=np.load(emb_file)
    with open(oov_fn) as f:
        for l in f:
            rec=l.rstrip().split(' ')
            if len(rec)==2: #skip the first line.
                continue 
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]]=np.array([float(r) for r in rec[1:] ])
    np.save(emb_file, embedding.astype('float32') )
    
parser = argparse.ArgumentParser()
parser.add_argument('--laptop_emb_np', type=str, default="laptop_emb.vec.npy")
parser.add_argument('--restaurant_emb_np', type=str, default="restaurant_emb.vec.npy")
parser.add_argument('--out_dir', type=str, default="data/prep_data/")
parser.add_argument('--laptop_oov', type=str, default="laptop_oov.vec")
parser.add_argument('--restaurant_oov', type=str, default="restaurant_oov.vec")
parser.add_argument('--word_idx', type=str, default="word_idx.json")
args = parser.parse_args()

fill_np_embedding(args.out_dir+args.laptop_emb_np, args.out_dir+args.word_idx, args.out_dir+args.laptop_oov)

fill_np_embedding(args.out_dir+args.restaurant_emb_np, args.out_dir+args.word_idx, args.out_dir+args.restaurant_oov)