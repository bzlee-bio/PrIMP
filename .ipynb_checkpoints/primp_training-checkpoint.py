import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
import argparse

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
    def w2vec(self, data, label):
        vec_data = np.empty((0,300))
        lab = []
        rm_list = []
        for _data, _label in zip(data, label):
            temp = np.zeros((1,300))
            tempList = []
            keyErrFlag = False
            for AA in list(_data):
                try:
                    tempList.append(self.vocab[AA.upper()])
                except KeyError:
                    kerErrFlag = True
                    pass
            if not keyErrFlag:
                temp[0, :len(tempList)]=np.array(tempList)
                vec_data = np.append(vec_data, temp,axis=0)
                lab.append(_label)

        return np.array(vec_data,dtype=np.int), np.array(lab)
    
class trainDataHandler(w2v):
    def __init__(self, dataPath, tgtList, valFold, batchSize=64):
        super().__init__()
        _dataTot = dict()
        self.trData = dict()
        self.valData = dict()
        self.testData = dict()
        for tgt in tgtList:
            tgtName = tgt.replace('.csv','')
            _path = f'{dataPath}/{tgt}'
            _dataTot[tgt]=pd.read_csv(_path)
            
            _data = _dataTot[tgt]
            _data['Label']=1
            _data.loc[_data.Type=='Negative','Label']=0
            # print(_data)
            trList = (_data['DataType']=='Train') & (_data['nFold']!=valFold) 
            valList = (_data['DataType']=='Train') & (_data['nFold']==valFold)
            testList = (_data['DataType']=='Test')
            # print(trList)
            # print(valList)
            # print(testList)
            
            # print(_data.loc[trList].Sequence.values)
            print(f'Target: {tgt}, =====================================')
            print('Train ---', 'Positive: ', np.sum(_data.loc[trList].Label.values), 'Negative: ',len(_data.loc[trList].Label.values)-np.sum(_data.loc[trList].Label.values))
            print('Valid ---', 'Positive: ', np.sum(_data.loc[valList].Label.values), 'Negative: ',len(_data.loc[valList].Label.values)-np.sum(_data.loc[valList].Label.values))            
            print('Test ---', 'Positive: ', np.sum(_data.loc[testList].Label.values), 'Negative: ',len(_data.loc[testList].Label.values)-np.sum(_data.loc[testList].Label.values))                        
            # print(_data.loc[trList].Label.values)
            self.trData[tgtName] = self.dataToDataset(self.w2vec(_data.loc[trList].Sequence.values, _data.loc[trList].Label.values), batchSize)
            self.valData[tgtName] = self.w2vec(_data.loc[valList].Sequence.values, _data.loc[valList].Label.values)
            self.testData[tgtName] = self.w2vec(_data.loc[testList].Sequence.values, _data.loc[testList].Label.values)

    def dataToDataset(self, _data, bSize):
        _x = _data[0]
        _y = _data[1]

        return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(_x),tf.data.Dataset.from_tensor_slices(_y))).shuffle(10000).batch(bSize)

    def tgtList(self):
        return list(self.trData.keys())
    
    
### Model generator
class Model(tf.keras.Model):
        
    def __init__(self, params):
        super(Model, self).__init__()
        self.regL1 = tf.keras.regularizers.l1(0.01)
        self.regL2 = tf.keras.regularizers.l2(0.01)
        self._embedding = tf.keras.layers.Embedding(input_dim = params['embInputDim'],output_dim=params['embOutputDim'],\
                                        mask_zero=True, name='EMBED')
        self._tgtSpecific = dict()

        ## Feature extracting layer
        for param in params['sharedLayer']:
            self._Shared = [self.modelUnitReturn(param, f'SL_{param[0]}_{i}') for i, param in enumerate(params['sharedLayer'])]
                                     
        for tgt in params['tgtList']:
            self._tgtSpecific[tgt] = [self.modelUnitReturn(param, f'TSL_{tgt}_{param[0]}_{i}') for i, param in enumerate(params['tgtSpecificLayer'])]
            self.params = params
        
    def modelUnitReturn(self, param, name):
        if param[0]=='conv1d':
            return tf.keras.layers.Conv1D(filters = param[1], kernel_size = param[2], strides = param[3], activation=param[4],
                                          kernel_regularizer=self.regL1, activity_regularizer=self.regL2, name=name, padding='valid')
        elif param[0]=='dropout':
            return tf.keras.layers.Dropout(param[1], name=name)
        elif param[0]=='mxpool1d':
            return tf.keras.layers.MaxPooling1D(pool_size = param[1], name=name)
        elif param[0]=='flatten':
            return tf.keras.layers.Flatten(name=name)
        elif param[0]=='biLSTM':
            return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(param[1], return_sequences=param[2],\
                                        dropout=param[3],\
                                        kernel_regularizer=self.regL1, activity_regularizer=self.regL2, name=name))
        elif param[0]=='dense':
            return tf.keras.layers.Dense(param[1], activation=param[2], name=name)
        elif param[0]=='bn':
            return tf.keras.layers.BatchNormalization()
        elif param[0]=='relu':
            return tf.keras.layers.ReLU()
        
    def model_build(self):
        ### Max Len: 300
        inp = tf.keras.Input(shape=(300))
        x = self._embedding(inp)
        for sharedLayer in self._Shared:
            x = sharedLayer(x)
        slOut = x
        tgtOutput = {}
        tgtModel = {}
        for tgt in self.params['tgtList']:
            for i, tSpecific in enumerate(self._tgtSpecific[tgt]):
                if i==0:
                    y = tSpecific(slOut)
                else:
                    y = tSpecific(y)
            tgtOutput[tgt] = y
            tgtModel[tgt] = tf.keras.Model(inp, tgtOutput[tgt])
        totOutput = tf.keras.layers.Concatenate(name='TotalOutput')(list(tgtOutput.values()))
        totModel = tf.keras.Model(inp, totOutput)
        sLayer = tf.keras.Model(inp, slOut)
        return tgtModel, totModel, sLayer
                
        
### Model trainer
class model_train():
    def __init__(self, params, dataHandler):
        self.trData = dataHandler.trData
        self.valData = dataHandler.valData
        self.testData = dataHandler.testData        
        M = Model(params)
        self.tgt_model, self.tot_model, self.sLayer = M.model_build()
        print(self.tot_model.summary())
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(params['learningRate'])
        self.params = params
        self.METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')]      
        
    def prediction_prob_out(self, data):
        return tf.keras.activations.sigmoid(self.tot_model.predict(data))

    def prediction_raw_out(self, data):
        return self.tot_model.predict(data)
    
    @tf.function
    def grad_calc(self, tgt, data):
        with tf.GradientTape() as tape:
            logit = self.tgt_model[tgt](data[0])
            loss = self.loss_fn(data[1], tf.squeeze(logit))
        grads = tape.gradient(loss, self.tgt_model[tgt].trainable_variables)
        return grads, loss

    def model_train(self):
        grads_dict = {}
        loss_dict = {}
        _dataset = dict()
        ### Data sample
        minIterNum = 9999
        for tgt in self.params['tgtList']:
            _dataset[tgt] = list(self.trData[tgt].take(params['iterNum']))
            loss_dict[tgt] = 0
            if minIterNum>=len(_dataset[tgt]):
                minIterNum = len(_dataset[tgt])
        
        for _iter in range(minIterNum):
            for tgt in self.params['tgtList']:
                grads, loss = self.grad_calc(tgt, _dataset[tgt][_iter])
                grads_dict[tgt]=grads
                loss_dict[tgt] += loss.numpy()
                _perfRes1D = dict()

            for gr_key in self.params['tgtList']:
                self.optimizer.apply_gradients(zip(grads_dict[gr_key], self.tgt_model[gr_key].trainable_variables))
                loss_dict[gr_key]/=minIterNum
    
        return loss_dict

    def performance(self, valid=True):
        perf_dict = {}

        if valid:
            dataset = self.valData
            isVal = 'Val'
        else:
            dataset = self.testData
            isVal = 'Test'

        for tgt in self.params['tgtList']:
            logit = self.tgt_model[tgt](dataset[tgt][0])
            pred = tf.squeeze(logit)>=0
            temp = self.return_metrics(dataset[tgt][1], pred)
            
            perf_dict.update({f'{isVal}_{tgt}_{key}': val for key, val in temp.items()})
            
        return perf_dict
    
    def return_metrics(self, true, pred):
        return_data = {}
        for m in self.METRICS:
            m.update_state(true, pred)
            return_data[m.name]=m.result().numpy()
            m.reset_states()
        return return_data
    
    def totModelSaver(self, save_dir, epoch):
        self.tot_model.save_weights(f'{save_dir}/totModel_epoch_{epoch:0>3}')
        
    def sLayerSaver(self, save_dir, epoch):
        self.sLayer.save_weights(f'{save_dir}/sLayer_epoch_{epoch:0>3}')
        
    def totModelLoader(self, load_file):
        load_status = self.tot_model.load_weights(load_file)
        print(load_status.assert_consumed())

    def sLayerLoader(self, load_file):
        load_status = self.sLayer.load_weights(load_file)
        print(load_status.assert_consumed())
                
            
            
            
            
            

            
parser = argparse.ArgumentParser(description = 'prIMP')
parser.add_argument('--tgtList', metavar='tgtList', type=str, help='Target Lists')
parser.add_argument('--valFold', metavar='valFold', type=int, help='5-fold cross-validation, Validation dataset fold num, 0~5')
parser.add_argument('--modelID', metavar='modelID', type=int, help='0~15')
parser.add_argument('--tL', metavar = 'tL', type=str, help='Transfer learning, True/False')
parser.add_argument('--c', metavar = 'c', type=str, help='Seq identity cut-off')

args = parser.parse_args()
predTgt = args.tgtList
nValFold = int(args.valFold)
seqI = args.c

if args.tL =='True':
    transferLearning = True
else:
    transferLearning = False
print('TRANSFER LEARNING -------------', transferLearning)
if predTgt == 'all':
    inpTgt = [f'sodium_c_{seqI}.csv',f'calcium_c_{seqI}.csv',f'nAChR_c_{seqI}.csv',f'potassium_c_{seqI}.csv']
    maxEpoch = 220
else:
    predTgt = predTgt.split('.')
    predTgt = f'{predTgt[0]}_c_{seqI}.csv'
    inpTgt = [predTgt]
    maxEpoch=300

dataHandler = trainDataHandler('./Data/',inpTgt,nValFold)

############## Model params
### CNN model
_params = dict()
_params[0] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
         'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
         'tgtSpecificLayer':[['conv1d',64,3,1,None],
                             ['bn'],
                             ['relu'],
                             ['dropout',0.5],
                             ['mxpool1d',2],
                             ['flatten',3],                             
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16C32-C64'
         }

_params[1] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
         'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
         'tgtSpecificLayer':[['conv1d',64,3,1,None],
                         ['bn'],
                         ['relu'],
                             ['dropout',0.5],
                             ['mxpool1d',2],
                            ['conv1d',128,3,1,None],
                         ['bn'],
                         ['relu'],
                             ['dropout',0.5],
                             ['mxpool1d',2],
                             ['flatten'],                             
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16C32-C64C128'
          
         }

_params[2] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
         'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
         'tgtSpecificLayer':[['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                             ['dropout',0.5],
                             ['mxpool1d',2],
                             ['flatten'],                             
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16-C32'          
          
         }
                         
_params[3] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
         'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
         'tgtSpecificLayer':[['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                             ['dropout',0.5],
                             ['mxpool1d',2],
                            ['conv1d',64,3,1,None],
                         ['bn'],
                         ['relu'],
                             ['dropout',0.5],
                             ['mxpool1d',2],
                             ['flatten'],                             
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16-C32C64'                    
         } 

############# LSTM model
_params[4] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',64,True,0.3]],
          'tgtSpecificLayer':[['biLSTM',32,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L64-L32'                    
         }



_params[5] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',64,True,0.3],['biLSTM',32,True,0.3]],
          'tgtSpecificLayer':[['biLSTM',16,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L64L32-L16'           
         }

_params[6] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',64,True,0.3],['biLSTM',32,True,0.3]],
          'tgtSpecificLayer':[['biLSTM',16,True,0.3], ['biLSTM',8,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L64L32-L16L8'           
         }

_params[7] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',64,True,0.3]],
          'tgtSpecificLayer':[['biLSTM',32,True,0.3], ['biLSTM',16,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L64-L32L16'           
         }

###### CNN-LSTM model
_params[8] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
          'tgtSpecificLayer':[['biLSTM',32,True,0.3], ['biLSTM',16,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16-L32L16'           
         }

_params[9] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
          'tgtSpecificLayer':[['biLSTM',32,True,0.3], ['biLSTM',16,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16C32-L32L16'
         }

_params[10] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],
          'tgtSpecificLayer':[['biLSTM',16,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16C32-L16'
         }

_params[11] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2]],

          'tgtSpecificLayer':[['biLSTM',16,False,0.3], 
                            ['flatten'],
                            ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'C16-L16'
         }

###### LSTM-CNN model
_params[12] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',16,True,0.3]],

          'tgtSpecificLayer':[['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['flatten'],
                         ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L16-C32'
         }

_params[13] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',16,True,0.3]],

          'tgtSpecificLayer':[['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],     
                         ['flatten'],
                         ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L16-C32C64'
         }
_params[14] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',32,True,0.3],
                         ['biLSTM',16,True,0.3]],

          'tgtSpecificLayer':[['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['flatten'],
                         ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L32L16-C32'
         }

_params[15] = {'tgtList':dataHandler.tgtList(),
          'embInputDim':22,
          'embOutputDim':10,
          'learningRate':1e-4,
          'iterNum':20,
          'maxEpoch':300,
          'sharedLayer': [['biLSTM',32,True,0.3],
                         ['biLSTM',16,True,0.3]],

          'tgtSpecificLayer':[['conv1d',16,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],
                         ['conv1d',32,3,1,None],
                         ['bn'],
                         ['relu'],
                         ['dropout',0.5],
                         ['mxpool1d',2],     
                         ['flatten'],
                         ['dense',1,None]],
          'dataset':predTgt.replace('.csv',''),
          'architecture':'L32L16-C16C32'
         }
                
            

############## Model params done
params = _params[args.modelID]

mt = model_train(params, dataHandler)
predTgt = predTgt.replace('.csv','')
_predTgt = predTgt.split('_')[0]
tgtPath = f'./modelWeights/{_predTgt}'
if not os.path.isdir(tgtPath):
    os.mkdir(tgtPath)

subDirPath = f'{predTgt}_model_{args.modelID:0>3}_fold_{nValFold}_numAug_0_TL_{transferLearning}_c_{seqI}'
weightsPath = f'{tgtPath}/{subDirPath}'
logsPath = f'{tgtPath}/{subDirPath}/logs'
if not os.path.isdir(weightsPath):
    os.mkdir(weightsPath)
    os.mkdir(logsPath)

print('weightsPath',weightsPath)
    
if transferLearning:
    DBAASPOptPath = f'./modelWeights/DBAASP/DBAASPoptModel/model_{args.modelID:0>3}'
    mt.sLayerLoader(DBAASPOptPath)

print(f'Tgt Path: {tgtPath}')
try:
    df = pd.read_csv(f'{weightsPath}/logs/log.csv')
    trainedLen = df.shape[0]
    print(f'Already {trainedLen} were trained!')
    mt.totModelLoader(f'{weightsPath}/totModel_epoch_{trainedLen-1:0>3}')
    print('Data Load, epoch: ', trainedLen-1)
except:
    trainedLen = 0
print(weightsPath)

if predTgt=='all':
    maxEpoch = 400
else:
    maxEpoch = 500
for epoch in range(trainedLen, maxEpoch):
    trLoss = mt.model_train()
    _trLoss = {f'Train_{key}_loss':val for key, val in trLoss.items()}
    totMetric = dict()
    totMetric.update(_trLoss)

    if epoch == 0:
        with open(f'{logsPath}/log.csv', 'w') as f:
            for key in totMetric.keys():
                f.write(f'{key},')
            f.write('\n')
    with open(f'{logsPath}/log.csv', 'a') as f:            
        for value in totMetric.values():
            f.write(f'{value},')
        f.write('\n')
        
    mt.totModelSaver(weightsPath, epoch)
    if predTgt=='DBAASP':
        mt.sLayerSaver(weightsPath, epoch)    
        
print('Done!')