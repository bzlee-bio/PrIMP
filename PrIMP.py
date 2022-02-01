import tensorflow as tf
from tensorflow import keras
from Bio import SeqIO
import numpy as np
import pandas as pd
import argparse
import logging

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class w2v:
    def __init__(self):
        self.vocab = {' ': 0,
                     'A': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'K': 9,
                     'L': 10,
                     'M': 11,
                     'N': 12,
                     'P': 13,
                     'Q': 14,
                     'R': 15,
                     'S': 16,
                     'T': 17,
                     'V': 18,
                     'W': 19,
                     'Y': 20,
                     'X': 21}
    def w2vec(self, data):
        vec_data = np.empty((0,300))
        rm_list = []
        for _data in data:
            temp = np.zeros((1,300))
            tempList = list()
            for AA in list(_data[1]):
                try:
                    tempList.append(self.vocab[AA.upper()])
                except KeyError:
                    tempList.append(21)
            temp[0, :len(tempList)]=np.array(tempList)
            vec_data = np.append(vec_data, temp,axis=0)
        return np.array(vec_data,dtype=np.int64)

def gen_logger():
    logger = logging.getLogger('PrIMP')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    
    #4 handler instance 생성
    console = logging.StreamHandler()
    # file_handler = logging.FileHandler(filename="test.log")
    
    #5 handler 별로 다른 level 설정
    console.setLevel(logging.INFO)
    # file_handler.setLevel(logging.DEBUG)

    #6 handler 출력 format 지정
    console.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    #7 logger에 handler 추가
    logger.addHandler(console)
    # logger.addHandler(file_handler)

    return logger

logger = gen_logger()



parser = argparse.ArgumentParser(description='Prediction of ion channel modulating-peptides')
parser.add_argument('--fasta', metavar='fasta', type=str, help='Input fasta file')
parser.add_argument('--output', metavar='output', type=str, help='Output file name')    
args = parser.parse_args()

### Load sequences from fasta file
inp_fasta = args.fasta
seq_data = list()
for l in SeqIO.parse(inp_fasta, 'fasta'):
    seq_data.append([l.id, str(l.seq)])
seq_len = len(seq_data)
logger.info(f'Total {seq_len} sequences were loaded...')
### sequence to token
w_to_v = w2v()

loaded_seq = w_to_v.w2vec(seq_data)

### Load model
logger.info('PrIMP model load...')
tgts = ['calcium' , 'nAChRs', 'potassium','sodium']
models = dict()
for tgt in tgts:
    models[tgt] = keras.models.load_model(f'./saved_model/{tgt}')
    
logger.info('PrIMP prediction start...')
pred_res = dict()
for tgt in tgts:
    pred_res[tgt] = tf.nn.sigmoid(models[tgt].predict(loaded_seq)).numpy().squeeze()
    
df_inpseq = pd.DataFrame(seq_data,columns=['ID','Sequence'])
df_predres = pd.DataFrame(pred_res)
df_predres.columns = columns = ['Pred_tgt_calcium_probability' , 'Pred_tgt_nAChRs_probability', 'Pred_tgt_potassium_probability','Pred_tgt_sodium_probability']
df_mod = pd.DataFrame(np.where(df_predres>=0.5,'Modulate', '-'),columns=['Pred_tgt_calcium' , 'Pred_tgt_nAChRs', 'Pred_tgt_potassium','Pred_tgt_sodium'])

df_res =pd.concat([df_inpseq,df_mod],axis=1)
df_res = pd.concat([df_res,df_predres],axis=1)

df_res.to_csv(args.output, sep=',')
logger.info('PrIMP prediction done...')
logger.info(f'Prediction results were saved into "{args.output}"')

