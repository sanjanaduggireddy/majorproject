Importing Packages¶
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
train='data/kddcup/kddcup.data_10_percent.gz'
test='data/kddcup/kddcup.testdata.unlabeled.gz'
test21='data/kddcup/nsl-kdd/KDDTest-21.txt'
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
df.to_csv('kdd_proc.csv')
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
79043/79043 [==============================] - 336s 4ms/step - loss: 0.0584 - accuracy: 0.9845 - val_loss: 0.0296 - val_accuracy: 0.9917
Epoch 2/20
79043/79043 [==============================] - 658s 8ms/step - loss: 0.0446 - accuracy: 0.9885 - val_loss: 0.0314 - val_accuracy: 0.9909
Epoch 3/20
79043/79043 [==============================] - 659s 8ms/step - loss: 0.0412 - accuracy: 0.9890 - val_loss: 0.0284 - val_accuracy: 0.9905
Epoch 4/20
79043/79043 [==============================] - 651s 8ms/step - loss: 0.0380 - accuracy: 0.9897 - val_loss: 0.0220 - val_accuracy: 0.9931
Epoch 5/20
79043/79043 [==============================] - 674s 9ms/step - loss: 0.0364 - accuracy: 0.9907 - val_loss: 0.0265 - val_accuracy: 0.9929
Epoch 6/20
79043/79043 [==============================] - 702s 9ms/step - loss: 0.0366 - accuracy: 0.9913 - val_loss: 0.0242 - val_accuracy: 0.9943
Epoch 7/20
79043/79043 [==============================] - 524s 7ms/step - loss: 0.0386 - accuracy: 0.9916 - val_loss: 0.0205 - val_accuracy: 0.9946
Epoch 8/20
79043/79043 [==============================] - 522s 7ms/step - loss: 0.0380 - accuracy: 0.9915 - val_loss: 0.0212 - val_accuracy: 0.9940
Epoch 9/20
79043/79043 [==============================] - 524s 7ms/step - loss: 0.0389 - accuracy: 0.9918 - val_loss: 0.0311 - val_accuracy: 0.9897
Epoch 10/20
79043/79043 [==============================] - 525s 7ms/step - loss: 0.0401 - accuracy: 0.9918 - val_loss: 0.0199 - val_accuracy: 0.9954
Epoch 11/20
79043/79043 [==============================] - 525s 7ms/step - loss: 0.0386 - accuracy: 0.9919 - val_loss: 0.0227 - val_accuracy: 0.9939
Epoch 12/20
79043/79043 [==============================] - 527s 7ms/step - loss: 0.0388 - accuracy: 0.9918 - val_loss: 0.0171 - val_accuracy: 0.9957
Epoch 13/20
79043/79043 [==============================] - 531s 7ms/step - loss: 0.0382 - accuracy: 0.9920 - val_loss: 0.0234 - val_accuracy: 0.9942
Epoch 14/20
79043/79043 [==============================] - 542s 7ms/step - loss: 0.0399 - accuracy: 0.9921 - val_loss: 0.0165 - val_accuracy: 0.9958
Epoch 15/20
79043/79043 [==============================] - 539s 7ms/step - loss: 0.0348 - accuracy: 0.9918 - val_loss: 0.0282 - val_accuracy: 0.9919
Epoch 16/20
79043/79043 [==============================] - 646s 8ms/step - loss: 0.0336 - accuracy: 0.9920 - val_loss: 0.0183 - val_accuracy: 0.9954
Epoch 17/20
79043/79043 [==============================] - 568s 7ms/step - loss: 0.0401 - accuracy: 0.9918 - val_loss: 0.0213 - val_accuracy: 0.9939
Epoch 18/20
79043/79043 [==============================] - 547s 7ms/step - loss: 0.0343 - accuracy: 0.9917 - val_loss: 0.0253 - val_accuracy: 0.9937
Epoch 19/20
79043/79043 [==============================] - 584s 7ms/step - loss: 0.0324 - accuracy: 0.9918 - val_loss: 0.0200 - val_accuracy: 0.9953
Epoch 20/20
79043/79043 [==============================] - 611s 8ms/step - loss: 0.0333 - accuracy: 0.9921 - val_loss: 0.0185 - val_accuracy: 0.9953
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

cnn_acc = accuracy_score(y_pred, y_test)
cnn_prec = precision_score(y_pred, y_test,average='weighted')
cnn_rec = recall_score(y_pred, y_test,average='weighted')
cnn_f1 = f1_score(y_pred, y_test,average='weighted')
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
79043/79043 [==============================] - 2017s 26ms/step - loss: 0.0126 - acc: 0.9727 - val_loss: 0.0073 - val_acc: 0.9793
Epoch 2/20
79043/79043 [==============================] - 2010s 25ms/step - loss: 0.0078 - acc: 0.9812 - val_loss: 0.0058 - val_acc: 0.9804
Epoch 3/20
79043/79043 [==============================] - 2011s 25ms/step - loss: 0.0070 - acc: 0.9819 - val_loss: 0.0052 - val_acc: 0.9826
Epoch 4/20
79043/79043 [==============================] - 2012s 25ms/step - loss: 0.0063 - acc: 0.9852 - val_loss: 0.0047 - val_acc: 0.9883
Epoch 5/20
79043/79043 [==============================] - 2009s 25ms/step - loss: 0.0057 - acc: 0.9881 - val_loss: 0.0040 - val_acc: 0.9885
Epoch 6/20
79043/79043 [==============================] - 2014s 25ms/step - loss: 0.0052 - acc: 0.9888 - val_loss: 0.0037 - val_acc: 0.9890
Epoch 7/20
79043/79043 [==============================] - 2011s 25ms/step - loss: 0.0049 - acc: 0.9891 - val_loss: 0.0044 - val_acc: 0.9896
Epoch 8/20
79043/79043 [==============================] - 2014s 25ms/step - loss: 0.0047 - acc: 0.9894 - val_loss: 0.0034 - val_acc: 0.9894
Epoch 9/20
79043/79043 [==============================] - 1994s 25ms/step - loss: 0.0046 - acc: 0.9895 - val_loss: 0.0039 - val_acc: 0.9899
Epoch 10/20
79043/79043 [==============================] - 1982s 25ms/step - loss: 0.0044 - acc: 0.9898 - val_loss: 0.0033 - val_acc: 0.9897
Epoch 11/20
79043/79043 [==============================] - 1977s 25ms/step - loss: 0.0043 - acc: 0.9899 - val_loss: 0.0032 - val_acc: 0.9898
Epoch 12/20
79043/79043 [==============================] - 1985s 25ms/step - loss: 0.0042 - acc: 0.9900 - val_loss: 0.0032 - val_acc: 0.9899
Epoch 13/20
79043/79043 [==============================] - 1563s 20ms/step - loss: 0.0041 - acc: 0.9902 - val_loss: 0.0030 - val_acc: 0.9904
Epoch 14/20
79043/79043 [==============================] - 1493s 19ms/step - loss: 0.0040 - acc: 0.9903 - val_loss: 0.0033 - val_acc: 0.9901
Epoch 15/20
79043/79043 [==============================] - 1492s 19ms/step - loss: 0.0040 - acc: 0.9904 - val_loss: 0.0029 - val_acc: 0.9903
Epoch 16/20
79043/79043 [==============================] - 1501s 19ms/step - loss: 0.0039 - acc: 0.9906 - val_loss: 0.0033 - val_acc: 0.9894
Epoch 17/20
79043/79043 [==============================] - 1498s 19ms/step - loss: 0.0038 - acc: 0.9907 - val_loss: 0.0036 - val_acc: 0.9908
Epoch 18/20
79043/79043 [==============================] - 1501s 19ms/step - loss: 0.0038 - acc: 0.9908 - val_loss: 0.0028 - val_acc: 0.9911
Epoch 19/20
79043/79043 [==============================] - 1488s 19ms/step - loss: 0.0037 - acc: 0.9909 - val_loss: 0.0028 - val_acc: 0.9910
Epoch 20/20
79043/79043 [==============================] - 1481s 19ms/step - loss: 0.0036 - acc: 0.9911 - val_loss: 0.0027 - val_acc: 0.9908
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
3088/3088 [==============================] - 15s 5ms/step
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
1000/1000 [==============================] - 54s 34ms/step - loss: 0.2085 - val_loss: 0.0824
Epoch 2/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0829 - val_loss: 0.0710
Epoch 3/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0734 - val_loss: 0.0693
Epoch 4/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0647 - val_loss: 0.0651
Epoch 5/20
1000/1000 [==============================] - 33s 34ms/step - loss: 0.0637 - val_loss: 0.0650
Epoch 6/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0694 - val_loss: 0.0642
Epoch 7/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0661 - val_loss: 0.0638
Epoch 8/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0556 - val_loss: 0.0659
Epoch 9/20
1000/1000 [==============================] - 33s 33ms/step - loss: 0.0535 - val_loss: 0.0652
Epoch 10/20
1000/1000 [==============================] - 35s 35ms/step - loss: 0.0613 - val_loss: 0.0636
Epoch 11/20
1000/1000 [==============================] - 35s 35ms/step - loss: 0.0643 - val_loss: 0.0655
Epoch 12/20
1000/1000 [==============================] - 35s 35ms/step - loss: 0.0595 - val_loss: 0.0641
Epoch 13/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0598 - val_loss: 0.0635
Epoch 14/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0600 - val_loss: 0.0636
Epoch 15/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0597 - val_loss: 0.0337
Epoch 16/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0315 - val_loss: 0.0312
Epoch 17/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0276 - val_loss: 0.0288
Epoch 18/20
1000/1000 [==============================] - 37s 37ms/step - loss: 0.0263 - val_loss: 0.0307
Epoch 19/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0309 - val_loss: 0.0306
Epoch 20/20
1000/1000 [==============================] - 36s 36ms/step - loss: 0.0304 - val_loss: 0.0308
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
1000/1000 [==============================] - 5s 5ms/step - loss: -18.8211 - accuracy: 0.4405
Epoch 2/20
1000/1000 [==============================] - 5s 5ms/step - loss: -137.0501 - accuracy: 0.4585
Epoch 3/20
1000/1000 [==============================] - 5s 5ms/step - loss: -402.5062 - accuracy: 0.3820
Epoch 4/20
1000/1000 [==============================] - 5s 5ms/step - loss: -810.2693 - accuracy: 0.3485
Epoch 5/20
1000/1000 [==============================] - 5s 5ms/step - loss: -1318.6145 - accuracy: 0.3575
Epoch 6/20
1000/1000 [==============================] - 6s 6ms/step - loss: -1827.8910 - accuracy: 0.3595
Epoch 7/20
1000/1000 [==============================] - 6s 6ms/step - loss: -2118.4368 - accuracy: 0.3985
Epoch 8/20
1000/1000 [==============================] - 6s 6ms/step - loss: -3175.8289 - accuracy: 0.3705
Epoch 9/20
1000/1000 [==============================] - 6s 6ms/step - loss: -3320.3140 - accuracy: 0.4100
Epoch 10/20
1000/1000 [==============================] - 6s 6ms/step - loss: -4449.7261 - accuracy: 0.3735
Epoch 11/20
1000/1000 [==============================] - 6s 6ms/step - loss: -5818.7524 - accuracy: 0.3545
Epoch 12/20
1000/1000 [==============================] - 6s 6ms/step - loss: -5829.9360 - accuracy: 0.4110
Epoch 13/20
1000/1000 [==============================] - 6s 6ms/step - loss: -7037.6611 - accuracy: 0.3775
Epoch 14/20
1000/1000 [==============================] - 6s 6ms/step - loss: -7993.5215 - accuracy: 0.4075
Epoch 15/20
1000/1000 [==============================] - 6s 6ms/step - loss: -8747.6289 - accuracy: 0.4280
Epoch 16/20
1000/1000 [==============================] - 6s 6ms/step - loss: -10856.0547 - accuracy: 0.3800
Epoch 17/20
1000/1000 [==============================] - 6s 6ms/step - loss: -12896.3271 - accuracy: 0.3680
Epoch 18/20
1000/1000 [==============================] - 6s 6ms/step - loss: -12767.6289 - accuracy: 0.3910
Epoch 19/20
1000/1000 [==============================] - 6s 6ms/step - loss: -13674.1494 - accuracy: 0.4220
Epoch 20/20
1000/1000 [==============================] - 6s 6ms/step - loss: -16615.0137 - accuracy: 0.3525
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
20874/20874 [==============================] - 871s 42ms/step - loss: 0.1599 - accuracy: 0.0432
Epoch 2/10
20874/20874 [==============================] - 859s 41ms/step - loss: 0.0334 - accuracy: 0.1067
Epoch 3/10
20874/20874 [==============================] - 859s 41ms/step - loss: 0.0112 - accuracy: 0.1172
Epoch 4/10
20874/20874 [==============================] - 856s 41ms/step - loss: 0.0052 - accuracy: 0.0979
Epoch 5/10
20874/20874 [==============================] - 871s 42ms/step - loss: 0.0027 - accuracy: 0.0794
Epoch 6/10
20874/20874 [==============================] - 858s 41ms/step - loss: 0.0017 - accuracy: 0.0605
Epoch 7/10
20874/20874 [==============================] - 859s 41ms/step - loss: 0.0012 - accuracy: 0.0479
Epoch 8/10
20874/20874 [==============================] - 858s 41ms/step - loss: 8.8019e-04 - accuracy: 0.0416
Epoch 9/10
20874/20874 [==============================] - 864s 41ms/step - loss: 6.7435e-04 - accuracy: 0.0340
Epoch 10/10
20874/20874 [==============================] - 864s 41ms/step - loss: 5.4784e-04 - accuracy: 0.0298
<keras.callbacks.History at 0x1f71838b7c8>
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
10000/10000 [==============================] - 235s 23ms/step - loss: -112.3969 - accuracy: 0.3071
Epoch 2/20
10000/10000 [==============================] - 235s 23ms/step - loss: -1305.5413 - accuracy: 0.2401
Epoch 3/20
10000/10000 [==============================] - 234s 23ms/step - loss: -3438.0595 - accuracy: 0.2341
Epoch 4/20
10000/10000 [==============================] - 235s 23ms/step - loss: -6665.9988 - accuracy: 0.2304
Epoch 5/20
10000/10000 [==============================] - 235s 23ms/step - loss: -11768.0653 - accuracy: 0.22474s - loss: -11734.0839 - ac - ETA: 3s - loss: -11740.9634 - accuracy: 
Epoch 6/20
10000/10000 [==============================] - 235s 24ms/step - loss: -18954.1331 - accuracy: 0.2270
Epoch 7/20
10000/10000 [==============================] - 235s 24ms/step - loss: -26433.6818 - accuracy: 0.21642s - lo
Epoch 8/20
10000/10000 [==============================] - 234s 23ms/step - loss: -35917.4854 - accuracy: 0.21994s - loss: -35871.6969 - accuracy: 0.220 - ETA: 4s - loss: -35 - ETA
Epoch 9/20
10000/10000 [==============================] - 233s 23ms/step - loss: -47496.7988 - accuracy: 0.2140
Epoch 10/20
10000/10000 [==============================] - 234s 23ms/step - loss: -57701.4757 - accuracy: 0.2148
Epoch 11/20
10000/10000 [==============================] - 235s 23ms/step - loss: -71990.5779 - accuracy: 0.2142
Epoch 12/20
10000/10000 [==============================] - 229s 23ms/step - loss: -85546.0236 - accuracy: 0.2111
Epoch 13/20
10000/10000 [==============================] - 229s 23ms/step - loss: -103993.8437 - accuracy: 0.2120
Epoch 14/20
10000/10000 [==============================] - 234s 23ms/step - loss: -122864.9538 - accuracy: 0.2165
Epoch 15/20
10000/10000 [==============================] - 235s 23ms/step - loss: -142507.1557 - accuracy: 0.2205
Epoch 16/20
10000/10000 [==============================] - 234s 23ms/step - loss: -162511.0392 - accuracy: 0.2181
Epoch 17/20
10000/10000 [==============================] - 236s 24ms/step - loss: -179695.6200 - accuracy: 0.2159
Epoch 18/20
10000/10000 [==============================] - 236s 24ms/step - loss: -207172.2352 - accuracy: 0.2029
Epoch 19/20
10000/10000 [==============================] - 237s 24ms/step - loss: -236123.5443 - accuracy: 0.2117
Epoch 20/20
10000/10000 [==============================] - 236s 24ms/step - loss: -252276.3599 - accuracy: 0.2056s - l
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
0	LSTM Autoencoder	0.971	0.996	0.995	0.996
1	DNN	0.793	1.000	0.793	0.885
2	Autoencoder DNN	0.737	0.793	0.737	0.760
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
