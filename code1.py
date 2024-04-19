Importing Package  
import warnings
warnings.filterwarnings('ignore')
# importing required libraries
import numpy as np
import pandas as pd
import pickle # saving and loading trained model
from os import path

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer

# importing library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.layers import Input
from keras.models import Model
# representation of model layers
from keras.utils.vis_utils import plot_model
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
1 Physical GPU, 2 Logical GPUs
Reading Data
feature=["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","target"]
train='data/nslkdd/kddcup.data_10_percent.gz'
test='data/nslkdd/kddcup.testdata.unlabeled.gz'
test21='data/nslkdd/nsl-kdd/KDDTest-21.txt'
train_data=pd.read_csv(train,names=feature)
# test_data=pd.read_csv(test,names=feature)
# test_21 = pd.read_csv(test21, names= feature)
train_data
duration	protocol_type	service	flag	src_bytes	dst_bytes	land	wrong_fragment	urgent	hot	...	dst_host_srv_count	dst_host_same_srv_rate	dst_host_diff_srv_rate	dst_host_same_src_port_rate	dst_host_srv_diff_host_rate	dst_host_serror_rate	dst_host_srv_serror_rate	dst_host_rerror_rate	dst_host_srv_rerror_rate	target
0	0	tcp	http	SF	181	5450	0	0	0	0	...	9	1.0	0.0	0.11	0.00	0.00	0.00	0.0	0.0	normal.
1	0	tcp	http	SF	239	486	0	0	0	0	...	19	1.0	0.0	0.05	0.00	0.00	0.00	0.0	0.0	normal.
2	0	tcp	http	SF	235	1337	0	0	0	0	...	29	1.0	0.0	0.03	0.00	0.00	0.00	0.0	0.0	normal.
3	0	tcp	http	SF	219	1337	0	0	0	0	...	39	1.0	0.0	0.03	0.00	0.00	0.00	0.0	0.0	normal.
4	0	tcp	http	SF	217	2032	0	0	0	0	...	49	1.0	0.0	0.02	0.00	0.00	0.00	0.0	0.0	normal.
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
494016	0	tcp	http	SF	310	1881	0	0	0	0	...	255	1.0	0.0	0.01	0.05	0.00	0.01	0.0	0.0	normal.
494017	0	tcp	http	SF	282	2286	0	0	0	0	...	255	1.0	0.0	0.17	0.05	0.00	0.01	0.0	0.0	normal.
494018	0	tcp	http	SF	203	1200	0	0	0	0	...	255	1.0	0.0	0.06	0.05	0.06	0.01	0.0	0.0	normal.
494019	0	tcp	http	SF	291	1200	0	0	0	0	...	255	1.0	0.0	0.04	0.05	0.04	0.01	0.0	0.0	normal.
494020	0	tcp	http	SF	219	1234	0	0	0	0	...	255	1.0	0.0	0.17	0.05	0.00	0.01	0.0	0.0	normal.
494021 rows × 42 columns

train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 494021 entries, 0 to 494020
Data columns (total 42 columns):
duration                       494021 non-null int64
protocol_type                  494021 non-null object
service                        494021 non-null object
flag                           494021 non-null object
src_bytes                      494021 non-null int64
dst_bytes                      494021 non-null int64
land                           494021 non-null int64
wrong_fragment                 494021 non-null int64
urgent                         494021 non-null int64
hot                            494021 non-null int64
num_failed_logins              494021 non-null int64
logged_in                      494021 non-null int64
num_compromised                494021 non-null int64
root_shell                     494021 non-null int64
su_attempted                   494021 non-null int64
num_root                       494021 non-null int64
num_file_creations             494021 non-null int64
num_shells                     494021 non-null int64
num_access_files               494021 non-null int64
num_outbound_cmds              494021 non-null int64
is_host_login                  494021 non-null int64
is_guest_login                 494021 non-null int64
count                          494021 non-null int64
srv_count                      494021 non-null int64
serror_rate                    494021 non-null float64
srv_serror_rate                494021 non-null float64
rerror_rate                    494021 non-null float64
srv_rerror_rate                494021 non-null float64
same_srv_rate                  494021 non-null float64
diff_srv_rate                  494021 non-null float64
srv_diff_host_rate             494021 non-null float64
dst_host_count                 494021 non-null int64
dst_host_srv_count             494021 non-null int64
dst_host_same_srv_rate         494021 non-null float64
dst_host_diff_srv_rate         494021 non-null float64
dst_host_same_src_port_rate    494021 non-null float64
dst_host_srv_diff_host_rate    494021 non-null float64
dst_host_serror_rate           494021 non-null float64
dst_host_srv_serror_rate       494021 non-null float64
dst_host_rerror_rate           494021 non-null float64
dst_host_srv_rerror_rate       494021 non-null float64
target                         494021 non-null object
dtypes: float64(15), int64(23), object(4)
memory usage: 158.3+ MB
train_data.describe().T
count	mean	std	min	25%	50%	75%	max
duration	494021.0	47.979302	707.746472	0.0	0.00	0.0	0.00	58329.0
src_bytes	494021.0	3025.610296	988218.101045	0.0	45.00	520.0	1032.00	693375640.0
dst_bytes	494021.0	868.532425	33040.001252	0.0	0.00	0.0	0.00	5155468.0
land	494021.0	0.000045	0.006673	0.0	0.00	0.0	0.00	1.0
wrong_fragment	494021.0	0.006433	0.134805	0.0	0.00	0.0	0.00	3.0
urgent	494021.0	0.000014	0.005510	0.0	0.00	0.0	0.00	3.0
hot	494021.0	0.034519	0.782103	0.0	0.00	0.0	0.00	30.0
num_failed_logins	494021.0	0.000152	0.015520	0.0	0.00	0.0	0.00	5.0
logged_in	494021.0	0.148247	0.355345	0.0	0.00	0.0	0.00	1.0
num_compromised	494021.0	0.010212	1.798326	0.0	0.00	0.0	0.00	884.0
root_shell	494021.0	0.000111	0.010551	0.0	0.00	0.0	0.00	1.0
su_attempted	494021.0	0.000036	0.007793	0.0	0.00	0.0	0.00	2.0
num_root	494021.0	0.011352	2.012718	0.0	0.00	0.0	0.00	993.0
num_file_creations	494021.0	0.001083	0.096416	0.0	0.00	0.0	0.00	28.0
num_shells	494021.0	0.000109	0.011020	0.0	0.00	0.0	0.00	2.0
num_access_files	494021.0	0.001008	0.036482	0.0	0.00	0.0	0.00	8.0
num_outbound_cmds	494021.0	0.000000	0.000000	0.0	0.00	0.0	0.00	0.0
is_host_login	494021.0	0.000000	0.000000	0.0	0.00	0.0	0.00	0.0
is_guest_login	494021.0	0.001387	0.037211	0.0	0.00	0.0	0.00	1.0
count	494021.0	332.285690	213.147412	0.0	117.00	510.0	511.00	511.0
srv_count	494021.0	292.906557	246.322817	0.0	10.00	510.0	511.00	511.0
serror_rate	494021.0	0.176687	0.380717	0.0	0.00	0.0	0.00	1.0
srv_serror_rate	494021.0	0.176609	0.381017	0.0	0.00	0.0	0.00	1.0
rerror_rate	494021.0	0.057433	0.231623	0.0	0.00	0.0	0.00	1.0
srv_rerror_rate	494021.0	0.057719	0.232147	0.0	0.00	0.0	0.00	1.0
same_srv_rate	494021.0	0.791547	0.388189	0.0	1.00	1.0	1.00	1.0
diff_srv_rate	494021.0	0.020982	0.082205	0.0	0.00	0.0	0.00	1.0
srv_diff_host_rate	494021.0	0.028997	0.142397	0.0	0.00	0.0	0.00	1.0
dst_host_count	494021.0	232.470778	64.745380	0.0	255.00	255.0	255.00	255.0
dst_host_srv_count	494021.0	188.665670	106.040437	0.0	46.00	255.0	255.00	255.0
dst_host_same_srv_rate	494021.0	0.753780	0.410781	0.0	0.41	1.0	1.00	1.0
dst_host_diff_srv_rate	494021.0	0.030906	0.109259	0.0	0.00	0.0	0.04	1.0
dst_host_same_src_port_rate	494021.0	0.601935	0.481309	0.0	0.00	1.0	1.00	1.0
dst_host_srv_diff_host_rate	494021.0	0.006684	0.042133	0.0	0.00	0.0	0.00	1.0
dst_host_serror_rate	494021.0	0.176754	0.380593	0.0	0.00	0.0	0.00	1.0
dst_host_srv_serror_rate	494021.0	0.176443	0.380919	0.0	0.00	0.0	0.00	1.0
dst_host_rerror_rate	494021.0	0.058118	0.230590	0.0	0.00	0.0	0.00	1.0
dst_host_srv_rerror_rate	494021.0	0.057412	0.230140	0.0	0.00	0.0	0.00	1.0
# number of attack labels 
train_data['target'].value_counts()
smurf.              280790
neptune.            107201
normal.              97278
back.                 2203
satan.                1589
ipsweep.              1247
portsweep.            1040
warezclient.          1020
teardrop.              979
pod.                   264
nmap.                  231
guess_passwd.           53
buffer_overflow.        30
land.                   21
warezmaster.            20
imap.                   12
rootkit.                10
loadmodule.              9
ftp_write.               8
multihop.                7
phf.                     4
perl.                    3
spy.                     2
Name: target, dtype: int64
Label
# number of attack labels 
train_data['target'].value_counts()
smurf.              280790
neptune.            107201
normal.              97278
back.                 2203
satan.                1589
ipsweep.              1247
portsweep.            1040
warezclient.          1020
teardrop.              979
pod.                   264
nmap.                  231
guess_passwd.           53
buffer_overflow.        30
land.                   21
warezmaster.            20
imap.                   12
rootkit.                10
loadmodule.              9
ftp_write.               8
multihop.                7
phf.                     4
perl.                    3
spy.                     2
Name: target, dtype: int64
# changing attack labels to their respective attack class
def change_label(df):
  df.target.replace(['back.','land.','neptune.','pod.','smurf.','teardrop.'],'Dos',inplace=True)
  df.target.replace(['guess_passwd.','imap.','ftp_write.','multihop.','phf.','spy.','warezclient.','warezmaster.'],'R2L',inplace=True)      
  df.target.replace(['ipsweep.','nmap.','portsweep.','satan.'],'Probe',inplace=True)
  df.target.replace(['buffer_overflow.','loadmodule.','perl.','rootkit.'],'U2R',inplace=True)
change_label(train_data)
# distribution of attack classes
train_data.target.value_counts()
Dos        391458
normal.     97278
Probe        4107
R2L          1126
U2R            52
Name: target, dtype: int64
# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = train_data.copy()
multi_label = pd.DataFrame(multi_data.target)
# using standard scaler for normalizing
std_scaler = StandardScaler()
def standardization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
  return df

numeric_col = multi_data.select_dtypes(include='number').columns
data = standardization(multi_data,numeric_col)
# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data['intrusion'] = enc_label
#y_mul = multi_data['intrusion']
multi_data
duration	protocol_type	service	flag	src_bytes	dst_bytes	land	wrong_fragment	urgent	hot	...	dst_host_same_srv_rate	dst_host_diff_srv_rate	dst_host_same_src_port_rate	dst_host_srv_diff_host_rate	dst_host_serror_rate	dst_host_srv_serror_rate	dst_host_rerror_rate	dst_host_srv_rerror_rate	target	intrusion
0	-0.067792	tcp	http	SF	-0.002879	0.138664	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.022077	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	normal.	4
1	-0.067792	tcp	http	SF	-0.002820	-0.011578	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.146737	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	normal.	4
2	-0.067792	tcp	http	SF	-0.002824	0.014179	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.188291	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	normal.	4
3	-0.067792	tcp	http	SF	-0.002840	0.014179	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.188291	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	normal.	4
4	-0.067792	tcp	http	SF	-0.002842	0.035214	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.209067	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	normal.	4
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
494016	-0.067792	tcp	http	SF	-0.002748	0.030644	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.229844	1.028094	-0.464418	-0.436950	-0.25204	-0.249464	normal.	4
494017	-0.067792	tcp	http	SF	-0.002776	0.042902	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-0.897417	1.028094	-0.464418	-0.436950	-0.25204	-0.249464	normal.	4
494018	-0.067792	tcp	http	SF	-0.002856	0.010032	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.125961	1.028094	-0.306769	-0.436950	-0.25204	-0.249464	normal.	4
494019	-0.067792	tcp	http	SF	-0.002767	0.010032	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-1.167514	1.028094	-0.359318	-0.436950	-0.25204	-0.249464	normal.	4
494020	-0.067792	tcp	http	SF	-0.002840	0.011061	-0.006673	-0.04772	-0.002571	-0.044136	...	0.599396	-0.282867	-0.897417	1.028094	-0.464418	-0.436950	-0.25204	-0.249464	normal.	4
494021 rows × 43 columns

multi_data.drop(labels= [ 'target'], axis=1, inplace=True)
multi_data
duration	protocol_type	service	flag	src_bytes	dst_bytes	land	wrong_fragment	urgent	hot	...	dst_host_srv_count	dst_host_same_srv_rate	dst_host_diff_srv_rate	dst_host_same_src_port_rate	dst_host_srv_diff_host_rate	dst_host_serror_rate	dst_host_srv_serror_rate	dst_host_rerror_rate	dst_host_srv_rerror_rate	intrusion
0	-0.067792	tcp	http	SF	-0.002879	0.138664	-0.006673	-0.04772	-0.002571	-0.044136	...	-1.694315	0.599396	-0.282867	-1.022077	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	4
1	-0.067792	tcp	http	SF	-0.002820	-0.011578	-0.006673	-0.04772	-0.002571	-0.044136	...	-1.600011	0.599396	-0.282867	-1.146737	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	4
2	-0.067792	tcp	http	SF	-0.002824	0.014179	-0.006673	-0.04772	-0.002571	-0.044136	...	-1.505707	0.599396	-0.282867	-1.188291	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	4
3	-0.067792	tcp	http	SF	-0.002840	0.014179	-0.006673	-0.04772	-0.002571	-0.044136	...	-1.411403	0.599396	-0.282867	-1.188291	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	4
4	-0.067792	tcp	http	SF	-0.002842	0.035214	-0.006673	-0.04772	-0.002571	-0.044136	...	-1.317100	0.599396	-0.282867	-1.209067	-0.158629	-0.464418	-0.463202	-0.25204	-0.249464	4
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
494016	-0.067792	tcp	http	SF	-0.002748	0.030644	-0.006673	-0.04772	-0.002571	-0.044136	...	0.625558	0.599396	-0.282867	-1.229844	1.028094	-0.464418	-0.436950	-0.25204	-0.249464	4
494017	-0.067792	tcp	http	SF	-0.002776	0.042902	-0.006673	-0.04772	-0.002571	-0.044136	...	0.625558	0.599396	-0.282867	-0.897417	1.028094	-0.464418	-0.436950	-0.25204	-0.249464	4
494018	-0.067792	tcp	http	SF	-0.002856	0.010032	-0.006673	-0.04772	-0.002571	-0.044136	...	0.625558	0.599396	-0.282867	-1.125961	1.028094	-0.306769	-0.436950	-0.25204	-0.249464	4
494019	-0.067792	tcp	http	SF	-0.002767	0.010032	-0.006673	-0.04772	-0.002571	-0.044136	...	0.625558	0.599396	-0.282867	-1.167514	1.028094	-0.359318	-0.436950	-0.25204	-0.249464	4
494020	-0.067792	tcp	http	SF	-0.002840	0.011061	-0.006673	-0.04772	-0.002571	-0.044136	...	0.625558	0.599396	-0.282867	-0.897417	1.028094	-0.464418	-0.436950	-0.25204	-0.249464	4
494021 rows × 42 columns

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
multi_data['protocol_type']= label_encoder.fit_transform(multi_data['protocol_type'])
multi_data['service']= label_encoder.fit_transform(multi_data['service'])
multi_data['flag']= label_encoder.fit_transform(multi_data['flag'])

multi_data['protocol_type'].unique()
multi_data['service'].unique()
multi_data['flag'].unique()
array([ 9,  6,  1,  7,  5,  8,  2,  4,  3,  0, 10], dtype=int64)
multi_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 494021 entries, 0 to 494020
Data columns (total 42 columns):
duration                       494021 non-null float64
protocol_type                  494021 non-null int32
service                        494021 non-null int32
flag                           494021 non-null int32
src_bytes                      494021 non-null float64
dst_bytes                      494021 non-null float64
land                           494021 non-null float64
wrong_fragment                 494021 non-null float64
urgent                         494021 non-null float64
hot                            494021 non-null float64
num_failed_logins              494021 non-null float64
logged_in                      494021 non-null float64
num_compromised                494021 non-null float64
root_shell                     494021 non-null float64
su_attempted                   494021 non-null float64
num_root                       494021 non-null float64
num_file_creations             494021 non-null float64
num_shells                     494021 non-null float64
num_access_files               494021 non-null float64
num_outbound_cmds              494021 non-null float64
is_host_login                  494021 non-null float64
is_guest_login                 494021 non-null float64
count                          494021 non-null float64
srv_count                      494021 non-null float64
serror_rate                    494021 non-null float64
srv_serror_rate                494021 non-null float64
rerror_rate                    494021 non-null float64
srv_rerror_rate                494021 non-null float64
same_srv_rate                  494021 non-null float64
diff_srv_rate                  494021 non-null float64
srv_diff_host_rate             494021 non-null float64
dst_host_count                 494021 non-null float64
dst_host_srv_count             494021 non-null float64
dst_host_same_srv_rate         494021 non-null float64
dst_host_diff_srv_rate         494021 non-null float64
dst_host_same_src_port_rate    494021 non-null float64
dst_host_srv_diff_host_rate    494021 non-null float64
dst_host_serror_rate           494021 non-null float64
dst_host_srv_serror_rate       494021 non-null float64
dst_host_rerror_rate           494021 non-null float64
dst_host_srv_rerror_rate       494021 non-null float64
intrusion                      494021 non-null int32
dtypes: float64(38), int32(4)
memory usage: 150.8 MB
Feature Selection
X = multi_data.drop(["intrusion"],axis =1)
y = multi_data["intrusion"]
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif

selector = SelectPercentile(mutual_info_classif, percentile=35)
X_reduced = selector.fit_transform(X, y)
X_reduced.shape
(494021, 14)
cols = selector.get_support(indices=True)
selected_columns = X.iloc[:,cols].columns.tolist()
selected_columns
['protocol_type',
 'service',
 'src_bytes',
 'dst_bytes',
 'logged_in',
 'count',
 'srv_count',
 'srv_diff_host_rate',
 'dst_host_count',
 'dst_host_srv_count',
 'dst_host_same_srv_rate',
 'dst_host_diff_srv_rate',
 'dst_host_same_src_port_rate',
 'dst_host_srv_diff_host_rate']
df = multi_data[['protocol_type',
                 'service',
                 'src_bytes',
                 'dst_bytes',
                 'logged_in',
                 'count',
                 'srv_count',
                 'srv_diff_host_rate',
                 'dst_host_count',
                 'dst_host_srv_count',
                 'dst_host_same_srv_rate',
                 'dst_host_diff_srv_rate',
                 'dst_host_same_src_port_rate',
                 'dst_host_srv_diff_host_rate',
                  'intrusion']]
df.to_csv('nsl_proc.csv')
# splitting the dataset 80% for training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y, test_size=0.20, random_state=42)
#X_train=X_train.values
#X_test=X_test.values

X_train = X_train.reshape(-1, X_train.shape[1],1)
X_test = X_test.reshape(-1, X_test.shape[1],1)

Y_train=to_categorical(y_train)
Y_test=to_categorical(y_test)
ML_Model = []
accuracy = []
precision = []
recall = []
f1score = []

#function to call for storing the results
def storeResults(model, a,b,c,d):
    ML_Model.append(model)
    accuracy.append(round(a, 3))
    precision.append(round(b, 3))
    recall.append(round(c, 3))
    f1score.append(round(d, 3))
CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

verbose, epoch, batch_size = 1, 100, 4
activationFunction='relu'

def CNN():
    
    cnnmodel = Sequential()
    cnnmodel.add(Conv1D(filters=128, kernel_size=2, activation='relu',input_shape=(X_train.shape[1],X_train.shape[2])))
    cnnmodel.add(MaxPooling1D(pool_size=2))
    cnnmodel.add(Dropout(rate=0.2))
    cnnmodel.add(Flatten())
    cnnmodel.add(Dense(5, activation='softmax'))
    cnnmodel.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    cnnmodel.summary()
    return cnnmodel

cnnmodel = CNN()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 13, 128)           384       
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 6, 128)            0         
_________________________________________________________________
dropout (Dropout)            (None, 6, 128)            0         
_________________________________________________________________
flatten (Flatten)            (None, 768)               0         
_________________________________________________________________
dense (Dense)                (None, 5)                 3845      
=================================================================
Total params: 4,229
Trainable params: 4,229
Non-trainable params: 0
_________________________________________________________________
modelhistory = cnnmodel.fit(X_train, Y_train, epochs=20, verbose=verbose, validation_split=0.2, batch_size = batch_size)
Epoch 1/20
79043/79043 [==============================] - 316s 4ms/step - loss: 0.0558 - accuracy: 0.9846 - val_loss: 0.0308 - val_accuracy: 0.9902
Epoch 2/20
79043/79043 [==============================] - 652s 8ms/step - loss: 0.0434 - accuracy: 0.9884 - val_loss: 0.0372 - val_accuracy: 0.9887
Epoch 3/20
79043/79043 [==============================] - 660s 8ms/step - loss: 0.0415 - accuracy: 0.9896 - val_loss: 0.0234 - val_accuracy: 0.9947
Epoch 4/20
79043/79043 [==============================] - 650s 8ms/step - loss: 0.0401 - accuracy: 0.9907 - val_loss: 0.0260 - val_accuracy: 0.9902
Epoch 5/20
79043/79043 [==============================] - 669s 8ms/step - loss: 0.0372 - accuracy: 0.9908 - val_loss: 0.0249 - val_accuracy: 0.9934
Epoch 6/20
79043/79043 [==============================] - 707s 9ms/step - loss: 0.0387 - accuracy: 0.9912 - val_loss: 0.0183 - val_accuracy: 0.9947
Epoch 7/20
79043/79043 [==============================] - 527s 7ms/step - loss: 0.0395 - accuracy: 0.9913 - val_loss: 0.0170 - val_accuracy: 0.9955
Epoch 8/20
79043/79043 [==============================] - 524s 7ms/step - loss: 0.0373 - accuracy: 0.9914 - val_loss: 0.0222 - val_accuracy: 0.9934
Epoch 9/20
79043/79043 [==============================] - 529s 7ms/step - loss: 0.0373 - accuracy: 0.9916 - val_loss: 0.0187 - val_accuracy: 0.9955
Epoch 10/20
79043/79043 [==============================] - 529s 7ms/step - loss: 0.0342 - accuracy: 0.9914 - val_loss: 0.0178 - val_accuracy: 0.9955
Epoch 11/20
79043/79043 [==============================] - 529s 7ms/step - loss: 0.0359 - accuracy: 0.9916 - val_loss: 0.0205 - val_accuracy: 0.9948
Epoch 12/20
79043/79043 [==============================] - 529s 7ms/step - loss: 0.0375 - accuracy: 0.9916 - val_loss: 0.0174 - val_accuracy: 0.9949
Epoch 13/20
79043/79043 [==============================] - 532s 7ms/step - loss: 0.0339 - accuracy: 0.9915 - val_loss: 0.0221 - val_accuracy: 0.9954
Epoch 14/20
79043/79043 [==============================] - 543s 7ms/step - loss: 0.0348 - accuracy: 0.9918 - val_loss: 0.0235 - val_accuracy: 0.9940
Epoch 15/20
79043/79043 [==============================] - 540s 7ms/step - loss: 0.0378 - accuracy: 0.9919 - val_loss: 0.0182 - val_accuracy: 0.9953
Epoch 16/20
79043/79043 [==============================] - 646s 8ms/step - loss: 0.0392 - accuracy: 0.9918 - val_loss: 0.0199 - val_accuracy: 0.9950
Epoch 17/20
79043/79043 [==============================] - 569s 7ms/step - loss: 0.0360 - accuracy: 0.9920 - val_loss: 0.0203 - val_accuracy: 0.9944
Epoch 18/20
79043/79043 [==============================] - 548s 7ms/step - loss: 0.0400 - accuracy: 0.9916 - val_loss: 0.0203 - val_accuracy: 0.9942
Epoch 19/20
79043/79043 [==============================] - 585s 7ms/step - loss: 0.0356 - accuracy: 0.9916 - val_loss: 0.0216 - val_accuracy: 0.9945
Epoch 20/20
79043/79043 [==============================] - 612s 8ms/step - loss: 0.0372 - accuracy: 0.9917 - val_loss: 0.0197 - val_accuracy: 0.9945
# Plot of accuracy vs epoch for train and test dataset
plt.plot(modelhistory.history['accuracy'])
plt.plot(modelhistory.history['val_accuracy'])
plt.title("Plot of accuracy vs epoch for train and test dataset")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Plot of loss vs epoch for train and test dataset
plt.plot(modelhistory.history['loss'])
plt.plot(modelhistory.history['val_loss'])
plt.title("Plot of loss vs epoch for train and test dataset")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

cnnpredictions = cnnmodel.predict(X_test, verbose=1)
cnn_predict=np.argmax(cnnpredictions,axis=1)

y_pred = cnnmodel.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred,axis=1)

#y_prob = cnnmodel.predict_proba(X_test)[:, 1]

dl_acc = accuracy_score(y_pred, y_test)
dl_prec = precision_score(y_pred, y_test,average='weighted')
dl_rec = recall_score(y_pred, y_test,average='weighted')
dl_f1 = f1_score(y_pred, y_test,average='weighted')
3088/3088 [==============================] - 5s 2ms/step
3088/3088 [==============================] - 5s 2ms/step
#storeResults('CNN',dl_acc,dl_prec,dl_rec,dl_f1)
CNN + LSTM
import tensorflow as tf
tf.keras.backend.clear_session()

model1 = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding="causal",activation="relu",input_shape=(X_train.shape[1],X_train.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(5)
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(5e-4,
                                                             decay_steps=1000000,
                                                             decay_rate=0.98,
                                                             staircase=False)

model1.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.8),
              metrics=['acc'])
model1.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 14, 64)            384       
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 13, 64)            0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 13, 32)            6176      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 12, 32)            0         
_________________________________________________________________
lstm (LSTM)                  (None, 12, 128)           82432     
_________________________________________________________________
flatten (Flatten)            (None, 1536)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               196736    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128      
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 165       
=================================================================
Total params: 290,021
Trainable params: 290,021
Non-trainable params: 0
_________________________________________________________________
modelhistory = model1.fit(X_train, Y_train, epochs=20, verbose=verbose, validation_split=0.2, batch_size = batch_size)
Epoch 1/20
79043/79043 [==============================] - 1998s 25ms/step - loss: 0.0140 - acc: 0.9676 - val_loss: 0.0078 - val_acc: 0.9793
Epoch 2/20
79043/79043 [==============================] - 2004s 25ms/step - loss: 0.0075 - acc: 0.9838 - val_loss: 0.0049 - val_acc: 0.9864
Epoch 3/20
79043/79043 [==============================] - 2002s 25ms/step - loss: 0.0065 - acc: 0.9865 - val_loss: 0.0044 - val_acc: 0.9870
Epoch 4/20
79043/79043 [==============================] - 2007s 25ms/step - loss: 0.0058 - acc: 0.9876 - val_loss: 0.0044 - val_acc: 0.9890
Epoch 5/20
79043/79043 [==============================] - 2001s 25ms/step - loss: 0.0055 - acc: 0.9883 - val_loss: 0.0040 - val_acc: 0.9881
Epoch 6/20
79043/79043 [==============================] - 2008s 25ms/step - loss: 0.0052 - acc: 0.9886 - val_loss: 0.0038 - val_acc: 0.9887
Epoch 7/20
79043/79043 [==============================] - 2000s 25ms/step - loss: 0.0050 - acc: 0.9889 - val_loss: 0.0037 - val_acc: 0.9894
Epoch 8/20
79043/79043 [==============================] - 2006s 25ms/step - loss: 0.0048 - acc: 0.9890 - val_loss: 0.0035 - val_acc: 0.9890
Epoch 9/20
79043/79043 [==============================] - 1991s 25ms/step - loss: 0.0047 - acc: 0.9892 - val_loss: 0.0035 - val_acc: 0.9890
Epoch 10/20
79043/79043 [==============================] - 1991s 25ms/step - loss: 0.0046 - acc: 0.9894 - val_loss: 0.0037 - val_acc: 0.9891
Epoch 11/20
79043/79043 [==============================] - 1978s 25ms/step - loss: 0.0044 - acc: 0.9896 - val_loss: 0.0035 - val_acc: 0.9896
Epoch 12/20
79043/79043 [==============================] - 1994s 25ms/step - loss: 0.0043 - acc: 0.9897 - val_loss: 0.0034 - val_acc: 0.9895
Epoch 13/20
79043/79043 [==============================] - 1579s 20ms/step - loss: 0.0042 - acc: 0.9899 - val_loss: 0.0032 - val_acc: 0.9896
Epoch 14/20
79043/79043 [==============================] - 1493s 19ms/step - loss: 0.0042 - acc: 0.9900 - val_loss: 0.0030 - val_acc: 0.9903
Epoch 15/20
79043/79043 [==============================] - 1496s 19ms/step - loss: 0.0041 - acc: 0.9902 - val_loss: 0.0041 - val_acc: 0.9877
Epoch 16/20
79043/79043 [==============================] - 1495s 19ms/step - loss: 0.0040 - acc: 0.9903 - val_loss: 0.0029 - val_acc: 0.9906
Epoch 17/20
79043/79043 [==============================] - 1497s 19ms/step - loss: 0.0039 - acc: 0.9906 - val_loss: 0.0028 - val_acc: 0.9906
Epoch 18/20
79043/79043 [==============================] - 1498s 19ms/step - loss: 0.0038 - acc: 0.9909 - val_loss: 0.0028 - val_acc: 0.9909
Epoch 19/20
79043/79043 [==============================] - 1485s 19ms/step - loss: 0.0038 - acc: 0.9910 - val_loss: 0.0026 - val_acc: 0.9912
Epoch 20/20
79043/79043 [==============================] - 1495s 19ms/step - loss: 0.0037 - acc: 0.9913 - val_loss: 0.0032 - val_acc: 0.9901
# Plot of accuracy vs epoch for train and test dataset
plt.plot(modelhistory.history['acc'])
plt.plot(modelhistory.history['val_acc'])
plt.title("Plot of accuracy vs epoch for train and test dataset")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Plot of loss vs epoch for train and test dataset
plt.plot(modelhistory.history['loss'])
plt.plot(modelhistory.history['val_loss'])
plt.title("Plot of loss vs epoch for train and test dataset")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


y_pred = model1.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred,axis=1)

#y_prob = cnnmodel.predict_proba(X_test)[:, 1]

ense_acc = accuracy_score(y_pred, y_test)
ense_prec = precision_score(y_pred, y_test,average='weighted')
ense_rec = recall_score(y_pred, y_test,average='weighted')
ense_f1 = f1_score(y_pred, y_test,average='weighted')
3088/3088 [==============================] - 21s 7ms/step
#storeResults('CNNLSTM-NIDS',ense_acc,ense_prec,ense_rec,ense_f1)
Transformer LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
%matplotlib inline

from numpy.random import seed

import tensorflow as tf

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split
x_train1, x_test1 = train_test_split(X_reduced, test_size = 0.2, random_state = 0)

y_train1, y_test1 = train_test_split(y, test_size = 0.2, random_state = 0)
y_test1.shape
(98805,)
# normalize the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train1)
x_test = scaler.transform(x_test1)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)
['scaler_data']
# reshape inputs for LSTM [samples, timesteps, features]
X_train1 = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
print("Training data shape:", X_train1.shape)
X_test1 = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
print("Test data shape:", X_test1.shape)
Training data shape: (395216, 1, 14)
Test data shape: (98805, 1, 14)
# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
# create the autoencoder model
model_tl = autoencoder_model(X_train1)
model_tl.compile(optimizer='adam', loss='mae')
model_tl.summary()
WARNING:tensorflow:Layer lstm will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
WARNING:tensorflow:Layer lstm_1 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
WARNING:tensorflow:Layer lstm_2 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
WARNING:tensorflow:Layer lstm_3 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 1, 14)]           0         
_________________________________________________________________
lstm (LSTM)                  (None, 1, 16)             1984      
_________________________________________________________________
lstm_1 (LSTM)                (None, 4)                 336       
_________________________________________________________________
repeat_vector (RepeatVector) (None, 1, 4)              0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 1, 4)              144       
_________________________________________________________________
lstm_3 (LSTM)                (None, 1, 16)             1344      
_________________________________________________________________
time_distributed (TimeDistri (None, 1, 14)             238       
=================================================================
Total params: 4,046
Trainable params: 4,046
Non-trainable params: 0
_________________________________________________________________
nb_epochs = 20
batch_size = 2
history = model_tl.fit(X_train1, X_train1, epochs=nb_epochs, batch_size=batch_size,validation_split=0.05,steps_per_epoch=1000, validation_steps=1000).history
Epoch 1/20
1000/1000 [==============================] - 51s 33ms/step - loss: 0.2475 - val_loss: 0.0826
Epoch 2/20
1000/1000 [==============================] - 30s 30ms/step - loss: 0.0793 - val_loss: 0.0723
Epoch 3/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0632 - val_loss: 0.0657
Epoch 4/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0601 - val_loss: 0.0642
Epoch 5/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0636 - val_loss: 0.0633
Epoch 6/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0629 - val_loss: 0.0647
Epoch 7/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0626 - val_loss: 0.0623
Epoch 8/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0479 - val_loss: 0.0329
Epoch 9/20
1000/1000 [==============================] - 34s 34ms/step - loss: 0.0342 - val_loss: 0.0306
Epoch 10/20
1000/1000 [==============================] - 32s 32ms/step - loss: 0.0295 - val_loss: 0.0301
Epoch 11/20
1000/1000 [==============================] - 34s 34ms/step - loss: 0.0285 - val_loss: 0.0294
Epoch 12/20
1000/1000 [==============================] - 34s 34ms/step - loss: 0.0275 - val_loss: 0.0295
Epoch 13/20
1000/1000 [==============================] - 35s 35ms/step - loss: 0.0284 - val_loss: 0.0288
Epoch 14/20
1000/1000 [==============================] - 35s 35ms/step - loss: 0.0276 - val_loss: 0.0282
Epoch 15/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0274 - val_loss: 0.0277
Epoch 16/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0282 - val_loss: 0.0282
Epoch 17/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0273 - val_loss: 0.0287
Epoch 18/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0279 - val_loss: 0.0284
Epoch 19/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0311 - val_loss: 0.0280
Epoch 20/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0256 - val_loss: 0.0278
# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

y_pred=model_tl.predict(X_test1)
tl_acc = 1 - history['loss'][19]
tl_prec = precision_score(y_pred, y_test,average='weighted')
tl_rec = recall_score(y_pred, y_test,average='weighted')
tl_f1 = f1_score(y_pred, y_test,average='weighted')
storeResults('LSTM Autoencoder',tl_acc,tl_prec,tl_rec,tl_f1)
DNN
from tensorflow.keras import models, layers, optimizers, regularizers
import numpy as np
import random
from sklearn import model_selection, preprocessing
import tensorflow as tf
from tqdm import tqdm
# splitting the dataset 80% for training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y, test_size=0.20, random_state=42)
x_train = preprocessing.normalize(X_train)
x_test = preprocessing.normalize(X_test)
hidden_units = 10     # how many neurons in the hidden layer
activation = 'relu'   # activation function for hidden layer
l2 = 0.01             # regularization - how much we penalize large parameter values
learning_rate = 0.01  # how big our steps are in gradient descent
epochs = 5            # how many epochs to train for
batch_size = 2       # how many samples to use for each gradient descent update
# create a sequential model
model2 = models.Sequential()

# add the hidden layer
model2.add(layers.Dense(input_dim=14,
                       units=hidden_units, 
                       activation=activation))

# add the output layer
model2.add(layers.Dense(input_dim=hidden_units,
                       units=1,
                       activation='sigmoid'))

# define our loss function and optimizer
model2.compile(loss='binary_crossentropy',
              # Adam is a kind of gradient descent
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model2.summary()
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 10)                150       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 161
Trainable params: 161
Non-trainable params: 0
_________________________________________________________________
history = model2.fit(x_train, y_train, epochs=20, batch_size=batch_size,steps_per_epoch=1000, validation_steps=1000)
Epoch 1/20
1000/1000 [==============================] - 6s 6ms/step - loss: -25.9204 - accuracy: 0.3525
Epoch 2/20
1000/1000 [==============================] - 6s 6ms/step - loss: -208.6465 - accuracy: 0.3525
Epoch 3/20
1000/1000 [==============================] - 6s 6ms/step - loss: -497.4223 - accuracy: 0.3775
Epoch 4/20
1000/1000 [==============================] - 6s 6ms/step - loss: -1000.8032 - accuracy: 0.3365
Epoch 5/20
1000/1000 [==============================] - 6s 6ms/step - loss: -1418.6417 - accuracy: 0.3855
Epoch 6/20
1000/1000 [==============================] - 6s 6ms/step - loss: -1880.3000 - accuracy: 0.4100
Epoch 7/20
1000/1000 [==============================] - 6s 6ms/step - loss: -3103.7075 - accuracy: 0.3180
Epoch 8/20
1000/1000 [==============================] - 6s 6ms/step - loss: -3378.0266 - accuracy: 0.3835
Epoch 9/20
1000/1000 [==============================] - 6s 6ms/step - loss: -4227.2300 - accuracy: 0.3875
Epoch 10/20
1000/1000 [==============================] - 6s 6ms/step - loss: -5044.4375 - accuracy: 0.3850
Epoch 11/20
1000/1000 [==============================] - 6s 6ms/step - loss: -5821.9297 - accuracy: 0.4095
Epoch 12/20
1000/1000 [==============================] - 6s 6ms/step - loss: -6950.0835 - accuracy: 0.3765
Epoch 13/20
1000/1000 [==============================] - 6s 6ms/step - loss: -7752.4414 - accuracy: 0.4205
Epoch 14/20
1000/1000 [==============================] - 6s 6ms/step - loss: -9313.1328 - accuracy: 0.4060
Epoch 15/20
1000/1000 [==============================] - 6s 6ms/step - loss: -10211.2119 - accuracy: 0.3980
Epoch 16/20
1000/1000 [==============================] - 5s 5ms/step - loss: -10435.5723 - accuracy: 0.4080
Epoch 17/20
1000/1000 [==============================] - 4s 4ms/step - loss: -13218.7803 - accuracy: 0.3830
Epoch 18/20
1000/1000 [==============================] - 6s 6ms/step - loss: -14325.4219 - accuracy: 0.3940
Epoch 19/20
1000/1000 [==============================] - 6s 6ms/step - loss: -16825.0723 - accuracy: 0.3775
Epoch 20/20
1000/1000 [==============================] - 6s 6ms/step - loss: -18090.6641 - accuracy: 0.3750
# Plot of accuracy vs epoch for train and test dataset
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title("Plot of accuracy vs epoch for train and test dataset")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Plot of loss vs epoch for train and test dataset
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title("Plot of loss vs epoch for train and test dataset")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


y_pred = model2.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred,axis=1)

#y_prob = cnnmodel.predict_proba(X_test)[:, 1]

dnn_acc = accuracy_score(y_pred, y_test)
dnn_prec = precision_score(y_pred, y_test,average='weighted')
dnn_rec = recall_score(y_pred, y_test,average='weighted')
dnn_f1 = f1_score(y_pred, y_test,average='weighted')
3088/3088 [==============================] - 3s 1ms/step
storeResults('DNN',dnn_acc,dnn_prec,dnn_rec,dnn_f1)
Transformer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras import regularizers
# splitting the dataset 80% for training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y, test_size=0.20, random_state=42)
train_min = X_train.min()
train_max = X_train.max()
x_train = (x_train - train_min) / (train_max - train_min)
x_test = (x_test - train_min) / (train_max - train_min)
positives = x_train[y_train == 1]
negatives = x_train[y_train == 0]
## input layer 
input_layer = Input(shape=negatives.shape[1:])

## encoding part
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dense(75, activation='tanh')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(50, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(25, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(7, activation='relu')(encoded)

## decoding part
decoded = Dense(7, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(25, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(50, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(75, activation='tanh')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(100, activation='tanh')(decoded)

## output layer
output_layer = Dense(negatives.shape[1], activation='relu')(decoded)
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse", metrics=['accuracy'])
autoencoder.fit(negatives, negatives, batch_size = 15, epochs = 10, shuffle = True)
Epoch 1/10
20874/20874 [==============================] - 875s 42ms/step - loss: 0.0926 - accuracy: 0.0247
Epoch 2/10
20874/20874 [==============================] - 859s 41ms/step - loss: 0.0108 - accuracy: 0.0672
Epoch 3/10
20874/20874 [==============================] - 859s 41ms/step - loss: 0.0046 - accuracy: 0.0759
Epoch 4/10
20874/20874 [==============================] - 858s 41ms/step - loss: 0.0024 - accuracy: 0.0569
Epoch 5/10
20874/20874 [==============================] - 872s 42ms/step - loss: 0.0016 - accuracy: 0.0390
Epoch 6/10
20874/20874 [==============================] - 857s 41ms/step - loss: 0.0011 - accuracy: 0.0319
Epoch 7/10
20874/20874 [==============================] - 858s 41ms/step - loss: 8.6767e-04 - accuracy: 0.0246
Epoch 8/10
20874/20874 [==============================] - 858s 41ms/step - loss: 7.3572e-04 - accuracy: 0.0198
Epoch 9/10
20874/20874 [==============================] - 866s 41ms/step - loss: 6.3421e-04 - accuracy: 0.0159
Epoch 10/10
20874/20874 [==============================] - 864s 41ms/step - loss: 5.6129e-04 - accuracy: 0.0141
<keras.callbacks.History at 0x246187dd0c8>
hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
hidden_representation.add(autoencoder.layers[3])
hidden_representation.add(autoencoder.layers[4])
hidden_representation.add(autoencoder.layers[5])
hidden_representation.add(autoencoder.layers[6])
hidden_representation.add(autoencoder.layers[7])
hidden_representation.add(autoencoder.layers[8])
hidden_representation.add(autoencoder.layers[9])
x_train_transformed = hidden_representation.predict(x_train)
x_test_transformed = hidden_representation.predict(x_test)
predictor = Sequential()
predictor.add(Dense(64, activation='relu', input_shape=x_train_transformed.shape[1:]))
predictor.add(BatchNormalization())
predictor.add(Dropout(0.25))
predictor.add(Dense(64, activation='relu'))
predictor.add(BatchNormalization())
predictor.add(Dense(64, activation='relu'))
predictor.add(BatchNormalization())
predictor.add(Dense(64, activation='tanh'))
predictor.add(BatchNormalization())
predictor.add(Dense(1, activation='sigmoid'))
predictor.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = predictor.fit(x_train_transformed, y_train, batch_size=2, epochs=20, shuffle=True,steps_per_epoch=10000, validation_steps=10000)
Epoch 1/20
10000/10000 [==============================] - 236s 23ms/step - loss: -111.9295 - accuracy: 0.3108
Epoch 2/20
10000/10000 [==============================] - 235s 23ms/step - loss: -1263.4057 - accuracy: 0.2186
Epoch 3/20
10000/10000 [==============================] - 234s 23ms/step - loss: -3644.3453 - accuracy: 0.2248
Epoch 4/20
10000/10000 [==============================] - 235s 23ms/step - loss: -8010.7095 - accuracy: 0.2195
Epoch 5/20
10000/10000 [==============================] - 234s 23ms/step - loss: -13266.5993 - accuracy: 0.2304
Epoch 6/20
10000/10000 [==============================] - 235s 23ms/step - loss: -19610.4302 - accuracy: 0.2069
Epoch 7/20
10000/10000 [==============================] - 235s 24ms/step - loss: -28600.7227 - accuracy: 0.2028
Epoch 8/20
10000/10000 [==============================] - 234s 23ms/step - loss: -37400.0990 - accuracy: 0.21026s - loss: -37298.8558 - accuracy
Epoch 9/20
10000/10000 [==============================] - 234s 23ms/step - loss: -50376.1917 - accuracy: 0.2158
Epoch 10/20
10000/10000 [==============================] - 234s 23ms/step - loss: -59878.7186 - accuracy: 0.20731s - loss: -59857.9343 - 
Epoch 11/20
10000/10000 [==============================] - 234s 23ms/step - loss: -72990.4760 - accuracy: 0.2174
Epoch 12/20
10000/10000 [==============================] - 226s 23ms/step - loss: -92052.0361 - accuracy: 0.2098
Epoch 13/20
10000/10000 [==============================] - 227s 23ms/step - loss: -106587.0290 - accuracy: 0.2075
Epoch 14/20
10000/10000 [==============================] - 234s 23ms/step - loss: -123370.3005 - accuracy: 0.1994
Epoch 15/20
10000/10000 [==============================] - 235s 24ms/step - loss: -141070.6737 - accuracy: 0.2004
Epoch 16/20
10000/10000 [==============================] - 234s 23ms/step - loss: -167764.7511 - accuracy: 0.2059
Epoch 17/20
10000/10000 [==============================] - 236s 24ms/step - loss: -191373.5472 - accuracy: 0.2087
Epoch 18/20
10000/10000 [==============================] - 236s 24ms/step - loss: -216707.4199 - accuracy: 0.2012
Epoch 19/20
10000/10000 [==============================] - 236s 24ms/step - loss: -242440.9747 - accuracy: 0.2028
Epoch 20/20
10000/10000 [==============================] - 232s 23ms/step - loss: -278806.8021 - accuracy: 0.1971s - loss: -278744.5025  - ETA: 1s - loss: -
y_predict = predictor.predict(x_test_transformed)
trans_acc = accuracy_score(y_test, y_predict >= 0.35)
trans_prec = precision_score(y_test, y_predict >= 0.35,average='weighted')
trans_rec = recall_score(y_test, y_predict >= 0.35,average='weighted')
trans_f1 = f1_score(y_test, y_predict >= 0.35,average='weighted')
storeResults('Autoencoder DNN',trans_acc,trans_prec,trans_rec,trans_f1)
Comparison
#creating dataframe
result = pd.DataFrame({ 'ML Model' : ML_Model,
                        'Accuracy' : accuracy,
                        'Precision': precision,
                        'Recall'   : recall,
                        'F1-Score': f1score
                      })
result
ML Model	Accuracy	Precision	Recall	F1-Score
0	LSTM Autoencoder	0.974	1.000	0.793	0.885
1	DNN	0.793	1.000	0.793	0.885
2	Autoencoder DNN	0.791	0.783	0.791	0.784
Graph
classifier = ML_Model
y_pos = np.arange(len(classifier))
Accuracy
import matplotlib.pyplot as plt2
plt2.barh(y_pos, accuracy, align='center', alpha=0.5,color='blue')
plt2.yticks(y_pos, classifier)
plt2.xlabel('Accuracy Score')
plt2.title('Classification Performance')
plt2.show()

Precision
plt2.barh(y_pos, precision, align='center', alpha=0.5,color='red')
plt2.yticks(y_pos, classifier)
plt2.xlabel('Precision Score')
plt2.title('Classification Performance')
plt2.show()

Recall
plt2.barh(y_pos, recall, align='center', alpha=0.5,color='cyan')
plt2.yticks(y_pos, classifier)
plt2.xlabel('Recall Score')
plt2.title('Classification Performance')
plt2.show()

F1 Score
plt2.barh(y_pos, f1score, align='center', alpha=0.5,color='magenta')
plt2.yticks(y_pos, classifier)
plt2.xlabel('F1 Score')
plt2.title('Classification Performance')
plt2.show()
