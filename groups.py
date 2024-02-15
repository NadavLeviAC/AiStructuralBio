



import numpy as np
import pandas as pd
from Levenshtein import distance
def compute_edit_distance(seqs: pd.Series) -> np.ndarray:
    results = np.zeros((len(seqs), len(seqs)))
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i == j or results[i][j] > 0:
                continue
            results[i][j] = results[j][i] = distance(a, b)
    return results

train = pd.read_csv('./train.csv')
print('Train shape:', train.shape )
print(train.head())
train['x'] = train.protein_sequence.str.len()
vc = train.x.value_counts()
print(vc.head())

train['group'] = -1
grp = 0

# MUTATION THRESHOLD
M_THRESHOLD = 10
# INSERTION DELETION THRESHOLD
D_THRESHOLD = 3

for k in range(len(vc)):
    c = vc.index[k]
    # SUBSET OF TRAIN DATA WITH SAME PROTEIN LENGTH PLUS MINUS D_THRESHOLD
    tmp = train.loc[(train.x>=c-D_THRESHOLD)&(train.x<=c+D_THRESHOLD)&(train.group==-1)]
    if len(tmp)<=1: break
    # COMPUTE LEVENSTEIN DISTANCE
    x = compute_edit_distance(tmp.protein_sequence)
    # COUNT HOW MANY MUTATIONS WE SEE
    mutation = []
    for kk in range(1, M_THRESHOLD + 1):
        mutation.append(len(np.unique(np.where(x == kk)[0])))
    # FIND RELATED ROWS IN TRAIN WITH M_THRESHOLD MUTATIONS OR LESS
    y = np.unique(np.where((x > 0) & (x <= M_THRESHOLD))[0])
    seen = []
    for j in y:
        if j in seen: continue
        i = np.where(np.array(x[j,]) <= M_THRESHOLD)[0]
        seen += list(i)
        idx = tmp.iloc[i].index
        train.loc[idx, 'group'] = grp
        grp += 1
    ct = vc.iloc[k]
    ct2 = len(tmp)
    # print(f'k={k} len={c} ct={ct} ct2={ct2} dist_ct={mutation}')
    # if k==9: break

for k in range(10):
    print('#'*25)
    print(f'### GROUP {k}')
    print('#'*25)
    print(train.loc[train.group==k])

train = train.drop('x',axis=1)
train.to_csv('train_with_groups.csv',index=False)
train.head()