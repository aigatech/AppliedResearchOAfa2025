# %% [markdown]
# # Anomaly Detection in Network Data using XGBoost
# *A compact, demo-friendly notebook for AI@GT.*
# 
# **Goal:** Train a fast, explainable binary classifier that flags network anomalies using the `abmallick/network-traffic-anomaly` dataset.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Features, Value
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# %% [markdown]
# # Load data and drop ID columns

# %%
features = Features({
    'Dst Port': Value('float64'),
    'Protocol': Value('float64'),
    'Timestamp': Value('string'),
    'Flow Duration': Value('float64'),
    'Tot Fwd Pkts': Value('float64'),
    'Tot Bwd Pkts': Value('float64'),
    'TotLen Fwd Pkts': Value('float64'),
    'TotLen Bwd Pkts': Value('float64'),
    'Fwd Pkt Len Max': Value('float64'),
    'Fwd Pkt Len Min': Value('float64'),
    'Fwd Pkt Len Mean': Value('float64'),
    'Fwd Pkt Len Std': Value('float64'),
    'Bwd Pkt Len Max': Value('float64'),
    'Bwd Pkt Len Min': Value('float64'),
    'Bwd Pkt Len Mean': Value('float64'),
    'Bwd Pkt Len Std': Value('float64'),
    'Flow Byts/s': Value('float64'),
    'Flow Pkts/s': Value('float64'),
    'Flow IAT Mean': Value('float64'),
    'Flow IAT Std': Value('float64'),
    'Flow IAT Max': Value('float64'),
    'Flow IAT Min': Value('float64'),
    'Fwd IAT Tot': Value('float64'),
    'Fwd IAT Mean': Value('float64'),
    'Fwd IAT Std': Value('float64'),
    'Fwd IAT Max': Value('float64'),
    'Fwd IAT Min': Value('float64'),
    'Bwd IAT Tot': Value('float64'),
    'Bwd IAT Mean': Value('float64'),
    'Bwd IAT Std': Value('float64'),
    'Bwd IAT Max': Value('float64'),
    'Bwd IAT Min': Value('float64'),
    'Fwd PSH Flags': Value('float64'),
    'Bwd PSH Flags': Value('float64'),
    'Fwd URG Flags': Value('float64'),
    'Bwd URG Flags': Value('float64'),
    'Fwd Header Len': Value('float64'),
    'Bwd Header Len': Value('float64'),
    'Fwd Pkts/s': Value('float64'),
    'Bwd Pkts/s': Value('float64'),
    'Pkt Len Min': Value('float64'),
    'Pkt Len Max': Value('float64'),
    'Pkt Len Mean': Value('float64'),
    'Pkt Len Std': Value('float64'),
    'Pkt Len Var': Value('float64'),
    'FIN Flag Cnt': Value('float64'),
    'SYN Flag Cnt': Value('float64'),
    'RST Flag Cnt': Value('float64'),
    'PSH Flag Cnt': Value('float64'),
    'ACK Flag Cnt': Value('float64'),
    'URG Flag Cnt': Value('float64'),
    'CWE Flag Count': Value('float64'),
    'ECE Flag Cnt': Value('float64'),
    'Down/Up Ratio': Value('float64'),
    'Pkt Size Avg': Value('float64'),
    'Fwd Seg Size Avg': Value('float64'),
    'Bwd Seg Size Avg': Value('float64'),
    'Fwd Byts/b Avg': Value('float64'),
    'Fwd Pkts/b Avg': Value('float64'),
    'Fwd Blk Rate Avg': Value('float64'),
    'Bwd Byts/b Avg': Value('float64'),
    'Bwd Pkts/b Avg': Value('float64'),
    'Bwd Blk Rate Avg': Value('float64'),
    'Subflow Fwd Pkts': Value('float64'),
    'Subflow Fwd Byts': Value('float64'),
    'Subflow Bwd Pkts': Value('float64'),
    'Subflow Bwd Byts': Value('float64'),
    'Init Fwd Win Byts': Value('float64'),
    'Init Bwd Win Byts': Value('float64'),
    'Fwd Act Data Pkts': Value('float64'),
    'Fwd Seg Size Min': Value('float64'),
    'Active Mean': Value('float64'),
    'Active Std': Value('float64'),
    'Active Max': Value('float64'),
    'Active Min': Value('float64'),
    'Idle Mean': Value('float64'),
    'Idle Std': Value('float64'),
    'Idle Max': Value('float64'),
    'Idle Min': Value('float64'),
    'Label': Value('string'),

    # The extra columns that caused the mismatch:
    'Flow ID': Value('float64'),
    'Src IP': Value('float64'),
    'Src Port': Value('float64'),
    'Dst IP': Value('float64'),
})


# Load the dataset with the specified features
ds = load_dataset("abmallick/network-traffic-anomaly", split="train", features=features)

# Prune unnecessary columns
drop_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']
ds = ds.remove_columns([c for c in drop_cols if c in ds.column_names])

# %% [markdown]
# ## Convert to Pandas dataframe for data manipulation

# %%
df = ds.to_pandas()
df = df.replace([np.inf, -np.inf], np.nan) 
df.head()

# %% [markdown]
# ### We can better understand the data by reviewing the Label column to see what classifications we have in our dataset.

# %%
df['Label'].value_counts()

# %% [markdown]
# Next we can setup a LabelEncoder to map the classes in our label column to integers for model training.

# %%
le = LabelEncoder()
le.fit(df['Label'])
print(le.classes_)

# %% [markdown]
# # Split data into training set and test set using a 75:25 ratio.

# %%
# capture the labels
labels = df['Label'].copy()

# convert labels to integers
integer_labels = le.transform(labels)
y = integer_labels

# drop the label column from the dataframe (Also timestamp for simplicity)
df = df.copy()
X = df.drop(columns=['Label', 'Timestamp'])

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25, 
                                                    random_state=42)

# %% [markdown]
# Lets confirm that the shape of our training data is what we expect it to be

# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %%
# store preprocessed data and label encoder for later use
preprocessed_data = {
    'x_train':x_train,
    'y_train':y_train,
    'x_test':x_test,
    'y_test':y_test,
    'le':le
}

# %% [markdown]
# ## Train a Binary Classifier
# We will want to convert to binary labels. If the label matches the class "Benign" then it is nominal, else it is considered an anomaly.

# %%
def convert_label_to_binary(label_encoder, labels):
    benign_idx = np.where(label_encoder.classes_ == 'Benign')[0][0]
    my_labels = labels.copy()
    my_labels[my_labels != benign_idx] = 1 
    my_labels[my_labels == benign_idx] = 0
    return my_labels

binary_y_train = convert_label_to_binary(le, y_train)
binary_y_test = convert_label_to_binary(le, y_test)

# check how many anomalies are in our labels
print('Number of anomalies in y_train: ', binary_y_train.sum())
print('Number of anomalies in y_test:  ', binary_y_test.sum())

# %% [markdown]
# # Hyperparameters

# %%
hyperparams = {
    'num_rounds':        10,
    'max_depth':         8,
    'max_leaves':        2**8,
    'alpha':             0.9,
    'eta':               0.1,
    'gamma':             0.1,
    'learning_rate':     0.1,
    'subsample':         1,
    'reg_lambda':        1,
    'scale_pos_weight':  2,
    'tree_method':       'hist',
    'objective':         'binary:logistic',
    'verbose':           True
}

# %% [markdown]
# # Binary Classification Model Training

# %%
x_train.head()

# %%
y_train[0:100]

# %%
dtrain = xgb.DMatrix(x_train, label=binary_y_train)
dtest = xgb.DMatrix(x_test, label=binary_y_test)
evals = [(dtest, 'test',), (dtrain, 'train')]

# %%
num_rounds = hyperparams['num_rounds']

# %%
model = xgb.train(hyperparams, dtrain, num_rounds, evals=evals)

# %%
threshold = .5
true_labels = binary_y_test.astype(int)
true_labels.sum()

# %%
preds = model.predict(dtest)
print(preds)

# %%
pred_labels = (preds > threshold).astype(int)
print(pred_labels)

# %%
pred_labels.sum()

# %% [markdown]
# # Model Evaluation: Area under the curve

# %%
# compute the auc
auc = roc_auc_score(true_labels, preds)
print(auc)

# %%
print ('Accuracy:', accuracy_score(true_labels, pred_labels))

# %% [markdown]
# # Model Evaluation: Confusion Matrix

# %%
results = confusion_matrix(true_labels, pred_labels) 

print ('Confusion Matrix :')

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


plot_confusion_matrix(results, ['Normal','Anomaly'])


