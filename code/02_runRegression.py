import sys
import pandas as pd
import numpy as np
import glob
import os

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import SReLU

from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)


seed = 7
np.random.seed(seed)

outputs = ['time']
fold = ['nfold']

PP_OUT_FLAG = True
LOG_FLAG = True

class Regression():

    def __init__(self, trainFilenames, testFilenames, inputTag):
        assert len(trainFilenames) == len(testFilenames)
        self.inputTag = inputTag
        for index in range(len(trainFilenames)):
            trainFilename = trainFilenames[index]
            testFilename = testFilenames[index]
            print trainFilename
            print testFilename
            self.fitModel(trainFilename, testFilename)

    def fitModel(self, trainFilename, testFilename):
        trainDF = pd.read_csv(trainFilename)
        testDF = pd.read_csv(testFilename)
        headers = trainDF.columns.values.tolist()
        inputs = list(set(headers) - set(outputs))
        inputs = list(set(inputs) - set(fold))
        inputs.sort()

        for output in outputs:
            trainX = trainDF.loc[:, inputs].as_matrix()
            testX = testDF.loc[:, inputs].as_matrix()
            trainY = trainDF[output].as_matrix()
            if True:
                preProcModelInput = preprocessing.MinMaxScaler()
                preProcModelInput.fit_transform(trainX)
                trainX = preProcModelInput.transform(trainX)
                testX = preProcModelInput.transform(testX)
                print 'train size:'
                print trainX.shape
                print 'test size:'
                print testX.shape
            if PP_OUT_FLAG:
                preProcModelOutput = preprocessing.MinMaxScaler()
                preProcModelOutput.fit_transform(trainY.reshape(-1, 1))
                trainY = preProcModelOutput.transform(trainY.reshape(-1, 1))
                trainY = np.squeeze(np.asarray(trainY))
                print trainY
            if LOG_FLAG:
                trainY = transformer.transform(trainY.reshape(-1, 1))
                trainY = np.squeeze(np.asarray(trainY))
                print trainY
            for key in ['dl']:
                testDataFrameCpy = testDF.copy()
                testDataFrameCpy['output'] = output
                testDataFrameCpy['tag'] = self.inputTag
                testDataFrameCpy['method'] = key
                testDataFrameCpy['obsr'] = testDF[output].values.tolist()
                outputFilename = os.path.basename(trainFilename).replace('train', 'pred_%s_%s' % (output, key))
                outputFilename = '../results/%s' % (outputFilename)
                modelFilename = outputFilename.replace('pred', 'model')
                modelFilename = modelFilename.replace('csv', 'txt')
                if key == 'dl':
                    ndim = trainX.shape[1]
                    model = Sequential()
                    model.add(Dense(10, input_dim=ndim, init='normal'))
                    model.add(SReLU())
                    model.add(Dense(5, init='normal'))
                    model.add(SReLU())
                    model.add(Dense(1, init='normal'))
                    model.add(SReLU())
                    model.compile(loss='mae', optimizer='nadam')
                    model.fit(trainX, trainY, nb_epoch=1000)
                    yhat = model.predict(testX)
                else:
                    # other learning algorithms; removed for dl benchmarking
                    pass
                if LOG_FLAG:
                    yhat = transformer.inverse_transform(yhat.reshape(-1, 1))
                if PP_OUT_FLAG:
                    yhat = preProcModelOutput.inverse_transform(yhat.reshape(-1, 1))
                predY = np.squeeze(np.asarray(yhat))
                header = 'pred'
                testDataFrameCpy[header] = predY
                testDataFrameCpy.to_csv(outputFilename, index=False)

if __name__ == '__main__':
    # print 'coming here'
    if len(sys.argv) == 2:
        inputTag = sys.argv[1]
    else:
        print 'prob tag not given'
        exit(0)

    print inputTag

    trainFilenames = []
    testFilenames = []
    pattern = '../folds/train-%s-*.csv' % inputTag
    trainFiles = glob.glob(pattern)

    print trainFiles

    for trainFilename in trainFiles:
        trainFilenames.append(trainFilename)
        testFilename = trainFilename.replace('train', 'test')
        testFilenames.append(testFilename)

    print trainFilenames
    print testFilenames
    Regression(trainFilenames, testFilenames, inputTag)
