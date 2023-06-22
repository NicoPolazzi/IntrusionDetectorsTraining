import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def _encode_text_dummy(df, name):
    
    names = []
    dummies = pd.get_dummies(df.loc[:,name])
    i = 0
    
    tmpL = []
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
        names.append(dummy_name)
        _x = [i, x]
        tmpL.append(_x)
        i += 1
    
    df.drop(name, axis=1, inplace=True)
    return names, tmpL


def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs"""
    y = df[target]
    x = df.drop(columns=target)
    return x, y

def removeB(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    
    for col in str_df:
        df[col] = str_df[col]
    
    return df

def get_Bank(PATH, seed, scale = True, show = False, anoIndex = 0, normIndex = 1):
    from scipy.io import arff
    
    data, _ = arff.loadarff(PATH)
    df = pd.DataFrame(data)
    df = removeB(df)
    
    discreteCol = df[df.columns.difference(['y'])].columns.tolist()
    columns = df.columns
    
    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['y'].copy()
    labels[labels != 'no'] = anoIndex # anomalous
    labels[labels == 'no'] = normIndex # normal

    df['y'] = labels
    normal = df[df['y'] == normIndex] 
    abnormal = df[df['y'] == anoIndex]
    
    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]


    train_size = 26383
    val_size = 2551

    train_set = normal[:train_size]
    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = 580
    test_size = 1740
    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))
    
    x_train, y_train = _to_xy(train_set, target='y')
    x_val, y_val = _to_xy(val_set, target='y')
    x_test, y_test = _to_xy(test_set, target='y')
    
    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')
    
    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp
        
    index = np.arange(0, len(columns)-len(discreteCol)-1)
    
    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()
    
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['discreteCol'] = discreteCol
    dataset['selectedColumns'] = selected_columns
    dataset['index'] = index
    
    return dataset

def get_KDDCUP99_REV(PATH, seed, scale = True, show = False):
    
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_hot_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label'] 
    
    df = pd.read_csv(PATH, header=None, names=columns)
    discreteCol = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']
    
    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t
        
    # Delete nepture, smurf
    df_neptune = df[df['label'] == 'neptune.']
    df_smurf = df[df['label'] == 'smurf.']
    df = df.loc[~df.index.isin(df_neptune.index)]
    df = df.loc[~df.index.isin(df_smurf.index)]

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0 # anomalous
    labels[labels == 'normal.'] = 1 # normal

    df['label'] = labels
    normal = df[df['label'] == 1] 
    abnormal = df[df['label'] == 0]
    
    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    test_size_ab = int(len(abnormal_1)*(3/4))
    val_size_ab = int(len(abnormal_1)*(1/4))

    test_size_n = int(.15 * (len(normal) + len(abnormal_1)) - test_size_ab + 1)
    val_size_n = int(.05 * (len(normal) + len(abnormal_1)) - val_size_ab + 1)

    train_size = int(len(normal) - val_size_n - test_size_n)+1

    train_set = normal[:train_size]
    val_normal = normal[train_size: train_size+val_size_n]
    test_normal = normal[train_size+val_size_n: ]

    val_abnormal = abnormal[:val_size_ab]
    test_abnormal = abnormal[val_size_ab:val_size_ab+test_size_ab]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')
    
    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == 1])} normal records and {len(x_train[y_train == 0])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == 1])} normal records and {len(x_val[y_val == 0])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == 1])} normal records and {len(x_test[y_test == 0])} abnormal records')

    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp
    
    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()
    
    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
        
        
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler
    
    return dataset

def get_NSLKDD(PATH_TRAIN, PATH_TEST, seed, mx = 0.889, mz = 0.028, my = 0.083, scale = True, show = False):
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_hot_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label', 'unknown']
    
    train = pd.read_csv(PATH_TRAIN, delimiter = ',', header = None, names = columns)
    test = pd.read_csv(PATH_TEST, delimiter = ',', header = None, names = columns)
    
    train.drop(columns = ['unknown'], inplace = True)
    test.drop(columns = ['unknown'], inplace = True)
    
    rest = set(train.columns) - set(test.columns)
    for i in rest:
        idx = train.columns.get_loc(i)
        test.insert(loc=idx, column=i, value=0)
    
    
    df = pd.concat((train, test))
    discreteCol = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']
    
    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()
    labels[labels != 'normal'] = 0 # anomalous
    labels[labels == 'normal'] = 1 # normal

    df['label'] = labels
    normal = df[df['label'] == 1] 
    abnormal = df[df['label'] == 0]
    
    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]
    
    train_set = normal[:int(mx*len(normal))]
    val_normal = normal[int(mx*len(normal)): int(mx*len(normal))+int(mz*len(normal))]
    test_normal = normal[int(mx*len(normal))+int(mz*len(normal)): ]
    
    val_abnormal = abnormal_1[:int(mz*len(normal))]
    test_abnormal = abnormal_1[int(mz*len(normal)):int(mz*len(normal))+int(my*len(normal))+1]
    
    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))
    
    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')
    
    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == 1])} normal records and {len(x_train[y_train == 0])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == 1])} normal records and {len(x_val[y_val == 0])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == 1])} normal records and {len(x_test[y_test == 0])} abnormal records')

    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp
    
    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()
    
    index = np.arange(0, len(columns)-len(discreteCol)-1)
    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
        
        
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler
    
    return dataset

def get_ADFA(PATH, seed, scale = True, show = True):
    columns = ['packets', 'bytes', 'Duration', 'label']
    df = pd.read_csv(PATH+'/ADFANet_Shuffled_LabelOK.csv')
    
    discreteCol = ['packets']#'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']
    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()
    labels[labels != 'normal'] = 1 # anomalous
    labels[labels == 'normal'] = 0 # normal
    df['label'] = labels
    
    normal = df[df['label'] == 0] 
    abnormal = df[df['label'] == 1]
    
    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

#    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal#[int(len(abnormal)*.5):]

    train_size = int(len(normal)*.6)

    train_set = normal[:train_size]

#    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size:]

    val_size = 0
    test_size = int(len(abnormal_2))
#    val_abnormal = abnormal_1
    test_abnormal = abnormal_2

#    val_set = val_abnormal
    test_set = pd.concat((test_normal, test_abnormal))
    
    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(train_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_2)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == 0])} normal records and {len(x_train[y_train == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == 0])} normal records and {len(x_val[y_val == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == 0])} normal records and {len(x_test[y_test == 1])} abnormal records')

    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy().astype(np.float32)
    x_val = x_val.to_numpy().astype(np.float32)
    x_test = x_test.to_numpy().astype(np.float32)

    index = np.arange(0, len(columns)-len(discreteCol)-1)


    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
#        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
    
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_train#.astype(np.float32)
    dataset['y_val'] = y_train#.astype(np.float32)
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    np.save('/home/polazzi/datasets/adfa/ARN_x_train.npy', dataset['x_train'])
    np.save('/home/polazzi/datasets/adfa/ARN_y_train.npy', dataset['y_train'])
    np.save('/home/polazzi/datasets/adfa/ARN_x_test.npy', dataset['x_test'])
    np.save('/home/polazzi/datasets/adfa/ARN_y_test.npy', dataset['y_test'])

    return dataset

def get_ADFA_Corrected(PATH, seed, scale = True, show = True):
    columns = ['packets', 'bytes', 'Duration', 'label', 'packetBis']

    np_train=np.load('/home/polazzi/datasets/adfa/train_X_no_attack.npy')
    np_train=np.hstack((np_train, np.zeros((np_train.shape[0],1))))
    np_test_x=np.load('/home/polazzi/datasets/adfa/test_X.npy')
    np_test_y=np.load('/home/polazzi/datasets/adfa/test_Y.npy')
    np_test=np.hstack((np_test_x, np_test_y))
    
    train_len=np_train.shape[0]
    test_len=np_test_x.shape[0]
    
    df1=pd.DataFrame(np_train, columns= ['packets', 'bytes', 'Duration', 'label'])
    df2=pd.DataFrame(np_test, columns= ['packets', 'bytes', 'Duration', 'label'])
    df=pd.concat([df1,df2]).reset_index(drop=True)
    df['packetBis'] = df.loc[:, 'packets']
    
    discreteCol = ['packets']#'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()

    train_size = train_len

    train_set = df[:train_size].reset_index(drop=True)

    test_set = df[train_size:].reset_index(drop=True)
    
    test_size = test_len
#    val_normal = normal[train_size: train_size+val_size]
#    val_abnormal = abnormal_1
#    test_abnormal = abnormal_2
    
    x_train, y_train = _to_xy(train_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')
    
    #non uso val, è solo un trucco per far tornare il codice
    val_size = train_size
    val_set = train_set
    x_val, y_val = x_train, y_train 
    
    if show:
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == 0])} normal records and {len(x_train[y_train == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == 0])} normal records and {len(x_val[y_val == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == 0])} normal records and {len(x_test[y_test == 1])} abnormal records')

    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

#    x_train.apply(pd.to_numeric)
#    x_test.apply(pd.to_numeric)    
#    x_val.apply(pd.to_numeric)

    x_train = x_train.to_numpy().astype(np.float32)
    x_val = x_train#.to_numpy().astype(np.float32)
    x_test = x_test.to_numpy().astype(np.float32)

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
#        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
        
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_train
    dataset['y_val'] = y_train
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    if scale:
        dataset['scaler'] = scaler
    else:
        dataset['scaler'] = None
        
    np.save('/home/polazzi/datasets/adfa/ARN_x_train.npy', dataset['x_train'])
    np.save('/home/polazzi/datasets/adfa/ARN_y_train.npy', dataset['y_train'])
    np.save('/home/polazzi/datasets/adfa/ARN_x_test.npy', dataset['x_test'])
    np.save('/home/polazzi/datasets/adfa/ARN_y_test.npy', dataset['y_test'])

    return dataset

def get_CICIDS(PATH, seed, scale = True, show = True):
    x_train=pd.read_pickle('/home/polazzi/datasets/cicids/train_X_no_attack_no_one_hot_encoding.pkl').reset_index(drop=True)
    y_train=np.load('/home/polazzi/datasets/cicids/train_Y_no_attack_no_one_hot_encoding.npy')
    y_train_df=pd.DataFrame(y_train, columns=['label']).reset_index(drop=True) 
    x_test=pd.read_pickle('/home/polazzi/datasets/cicids/test_X_no_one_hot_encoding.pkl').reset_index(drop=True) 
    y_test=np.load('/home/polazzi/datasets/cicids/test_Y_no_one_hot_encoding.npy')
    y_test_df=pd.DataFrame(y_test, columns=['label']).reset_index(drop=True) 
       
        
    columns = list(x_train.columns)

    discreteCol = ['Protocol','Fwd_PSH_Flags','Bwd_PSH_Flags','Fwd_URG_Flags', 'Bwd_URG_Flags',
                    'FIN_Flag_Cnt','SYN_Flag_Cnt','RST_Flag_Cnt','PSH_Flag_Cnt','ACK_Flag_Cnt',
                    'URG_Flag_Cnt','CWE_Flag_Count','ECE_Flag_Cnt','DownUp_Ratio','Fwd_Bytsb_Avg',
                    'Fwd_Pktsb_Avg','Fwd_Blk_Rate_Avg','Bwd_Bytsb_Avg','Bwd_Pktsb_Avg','Bwd_Blk_Rate_Avg']


    train=pd.concat([x_train, y_train_df], axis=1)
    test=pd.concat([x_test, y_test_df], axis=1)
        
#    np_train=np.load('/home/polazzi/datasets/adfa/train_X_no_attack.npy')
#    np_train=np.hstack((np_train, np.zeros((np_train.shape[0],1))))
#    np_test_x=np.load('/home/polazzi/datasets/adfa/test_X.npy')
#    np_test_y=np.load('/home/polazzi/datasets/adfa/test_Y.npy')
#    np_test=np.hstack((np_test_x, np_test_y))

    train_len=train.shape[0]
    test_len=test.shape[0]

    df1=train #pd.DataFrame(np_train, columns= ['packets', 'bytes', 'Duration', 'label'])
    df2= test #pd.DataFrame(np_test, columns= ['packets', 'bytes', 'Duration', 'label'])
    df=pd.concat([df1,df2])
    
    #discreteCol = ['packets']#'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()
#    labels[labels != 'normal'] = 1 # anomalous
#    labels[labels == 'normal'] = 0 # normal
#    df['label'] = labels
    
#    normal = df[df['label'] == 0] 
#    abnormal = df[df['label'] == 1]
    
#    normal = shuffle(normal, random_state = seed)
#    abnormal = shuffle(abnormal, random_state = seed)

#    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
#    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    train_size = train_len

    train_set = df[:train_size]
#    val_normal = normal[train_size: train_size+val_size]
    test_set = df[train_size:]

    test_size = test_len
#    val_abnormal = abnormal_1
#    test_abnormal = abnormal_2

    #non uso val, è solo un trucco per far tornare il codice
    val_size = train_size
    val_set = train_set
#    test_set = pd.concat((test_normal, test_abnormal))
    
    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == 0])} normal records and {len(x_train[y_train == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == 0])} normal records and {len(x_val[y_val == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == 0])} normal records and {len(x_test[y_test == 1])} abnormal records')

    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

#    x_train.apply(pd.to_numeric)
#    x_test.apply(pd.to_numeric)    
#    x_val.apply(pd.to_numeric)

    x_train = x_train.to_numpy().astype(np.float32)
    x_val = x_val.to_numpy().astype(np.float32)
    x_test = x_test.to_numpy().astype(np.float32)

    index = np.arange(0, len(columns)-len(discreteCol)-1)


    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
        
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    np.save('/home/polazzi/datasets/cicids/ARN_x_train.npy', dataset['x_train'])
    np.save('/home/polazzi/datasets/cicids/ARN_y_train.npy', dataset['y_train'])
    np.save('/home/polazzi/datasets/cicids/ARN_x_test.npy', dataset['x_test'])
    np.save('/home/polazzi/datasets/cicids/ARN_y_test.npy', dataset['y_test'])

    return dataset
    

def get_ADFA_Corrected2(PATH, seed, scale = True, show = True):
    columns = ['packets', 'bytes', 'Duration', 'label', 'packetBis']

    df = pd.read_csv(PATH+'ADFANet_Shuffled_LabelOK.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    
    np_train=np.load('/home/polazzi/datasets/adfa/train_X_no_attack.npy')
    np_train=np.hstack((np_train, np.zeros((np_train.shape[0],1))))
    np_test_x=np.load('/home/polazzi/datasets/adfa/test_X.npy')
    np_test_y=np.load('/home/polazzi/datasets/adfa/test_Y.npy')
    np_test=np.hstack((np_test_x, np_test_y))
    
    train_len=np_train.shape[0]
    test_len=np_test_x.shape[0]
    
    df['packetBis'] = df.loc[:, 'packets']
    
    discreteCol = ['packets']#'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    labels = df['label'].copy()
    labels[labels != 'normal'] = 1 # anomalous
    labels[labels == 'normal'] = 0 # normal
    df['label'] = labels
    
    
    
    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encode_text_dummy(df, name)
        names.extend(n)
        oneHot[name] = t

    labels = df['label'].copy()

    train_size = train_len
    test_size = test_len

    train_set = df[:train_len].reset_index(drop=True)
    test_set = df[train_len:test_len].reset_index(drop=True)
    
#    val_normal = normal[train_size: train_size+val_size]
#    val_abnormal = abnormal_1
#    test_abnormal = abnormal_2
    
    x_train, y_train = _to_xy(train_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')
    
    #non uso val, è solo un trucco per far tornare il codice
    val_size = train_size
    val_set = train_set
    x_val, y_val = x_train, y_train 
    
    if show:
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == 0])} normal records and {len(x_train[y_train == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == 0])} normal records and {len(x_val[y_val == 1])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == 0])} normal records and {len(x_test[y_test == 1])} abnormal records')

    selected_columns = dict()
    
    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

#    x_train.apply(pd.to_numeric)
#    x_test.apply(pd.to_numeric)    
#    x_val.apply(pd.to_numeric)

    x_train = x_train.to_numpy().astype(np.float32)
    x_val = x_train#.to_numpy().astype(np.float32)
    x_test = x_test.to_numpy().astype(np.float32)

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
#        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
        
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    
    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)
    
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    
    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    if scale:
        dataset['scaler'] = scaler
    else:
        dataset['scaler'] = None
        

    np.save('/home/polazzi/datasets/adfa/ARN_x_train.npy', dataset['x_train'])
    np.save('/home/polazzi/datasets/adfa/ARN_y_train.npy', dataset['y_train'])
    np.save('/home/polazzi/datasets/adfa/ARN_x_test.npy', dataset['x_test'])
    np.save('/home/polazzi/datasets/adfa/ARN_y_test.npy', dataset['y_test'])

    return dataset
    
