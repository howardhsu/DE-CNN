import argparse
import numpy as np
import json
import subprocess

def gen_np_embedding(fn, word_idx_fn, out_fn, dim=300):
    with open(word_idx_fn) as f:
        word_idx=json.load(f)
    embedding=np.zeros((len(word_idx)+2, dim) )
    with open(fn) as f:
        for l in f:
            rec=l.rstrip().split(' ')
            if len(rec)==2: #skip the first line.
                continue 
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]]=np.array([float(r) for r in rec[1:] ])
    with open(out_fn+".oov.txt", "w") as fw:
        for w in word_idx:
            if embedding[word_idx[w] ].sum()==0.:
                fw.write(w+"\n")
    np.save(out_fn+".npy", embedding.astype('float32') )
    
parser = argparse.ArgumentParser()
parser.add_argument('--emb_dir', type=str, default="data/embedding/")
parser.add_argument('--out_dir', type=str, default="data/prep_data/")
parser.add_argument('--gen_emb', type=str, default="gen.vec")
parser.add_argument('--laptop_emb', type=str, default="laptop_emb.vec")
parser.add_argument('--restaurant_emb', type=str, default="restaurant_emb.vec")
parser.add_argument('--word_idx', type=str, default="word_idx.json")
parser.add_argument('--gen_dim', type=int, default=300)
parser.add_argument('--domain_dim', type=int, default=100)
args = parser.parse_args()

gen_np_embedding(args.emb_dir+args.gen_emb, args.out_dir+args.word_idx, args.out_dir+args.gen_emb, args.gen_dim)

gen_np_embedding(args.emb_dir+args.laptop_emb, args.out_dir+args.word_idx, args.out_dir+args.laptop_emb, args.domain_dim)

gen_np_embedding(args.emb_dir+args.restaurant_emb, args.out_dir+args.word_idx, args.out_dir+args.restaurant_emb, args.domain_dim)