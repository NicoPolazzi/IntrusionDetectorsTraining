import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def get_train(*args):
    """Get training dataset for ADFA"""
    return _get_adapted_dataset("train")

def get_test(*args):
    """Get testing dataset for ADFA"""
    return _get_adapted_dataset("test")

def get_shape_input():
    """Get shape of the dataset for ADFA"""
    return (None, 3)

def get_shape_label():
    """Get shape of the labels in ADFA"""
    return (None,)

def _get_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 3)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 3)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    
    
    col_names = _col_names()
    
#    df = pd.read_csv("/home/notebook/attack_generation/eGAN/Efficient-GAN-Anomaly-Detection/data/ADFANet_Shuffled_LabelOK.csv")
    ADFANet_train_X_no_attack=np.load('/home/polazzi/datasets/adfa/train_X_no_attack.npy' )
    ADFANet_test_X= np.load('/home/polazzi/datasets/adfa/test_X.npy')
    ADFANet_test_Y=np.load('/home/polazzi/datasets/adfa/test_Y.npy' )
    
    #df_train
    df1=pd.DataFrame(ADFANet_train_X_no_attack, columns= ["packets","bytes","Duration"])
    df2=pd.DataFrame(np.zeros((ADFANet_train_X_no_attack.shape[0], 1)), columns=["label"])
    frame=[df1, df2]
    df_train=pd.concat(frame, axis=1)

    #df_train
    df1=pd.DataFrame(ADFANet_test_X, columns= ["packets","bytes","Duration"])
    df2=pd.DataFrame(ADFANet_test_Y, columns=["label"])
    frame=[df1, df2]
    df_test=pd.concat(frame, axis=1)

        
    text_l = [] #no textual fields

    for name in text_l:
        _encode_text_dummy(df, name)

#    labels = df['label'].copy()
    #normal must be 0, anomaly 1
#    labels[labels != 'normal'] = 1
#    labels[labels == 'normal'] = 0
#    df['label'] = labels
    
    x_train, y_train = _to_xy(df_train, target='label')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

#    scaler = MinMaxScaler()
#    scaler.fit(x_train)
#    scaler.transform(x_train)
#    scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

def _get_adapted_dataset(split):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    dummies = df[target]
    result=df.drop(columns=[target])
    return np.array(result), dummies.to_numpy()

def _col_names():
    """Column names of the dataframe"""
    return ["packets",
            "bytes",
            "Duration",
            "label"]
        

def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho))

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy
