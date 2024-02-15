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

VER = 2
train = pd.read_csv('./train_with_groups.csv')
train = train.loc[train.group!=-1]
train.pH = train.pH.fillna(-1)
train['len'] = train.protein_sequence.str.len()
print( train.shape )
train.head()

vc = train.group.value_counts()
print('Number of groups',len(vc))
print('Group number and their counts:')
vc.head()


rows = []

print(f'Processing {len(vc)-1} groups...')
for k in range(len(vc)-1):
    print('#'*25)
    ct = vc.iloc[k]
    print('### group',k+1,'with ct=',ct)
    print('#'*25)

    # COMPUTE LEVENSHTEIN DISTANCE WITHIN GROUPS
    grp = vc.index[k]
    tmp = train.loc[train.group==grp]
    # d = tmp.protein_sequence.str.edit_distance_matrix()
    # d = np.array( d.to_pandas().values.tolist() )
    d = compute_edit_distance(tmp.protein_sequence)

    # d = np.array(d.to_pandas().values.tolist())

    idx_x = np.where( d==1 )[0]
    print(idx_x)
    idx_y = np.where( d==1 )[1]

    # FIND PAIRS WITHIN GROUPS
    print(f'Processing {len(idx_x)} pairs...')
    for i,(x,y) in enumerate(zip(idx_x,idx_y)):
        if i%100==0: print(i,', ',end='')
        # DONT DOUBLE COUNT PAIRS
        if y <= x: continue
        row1 = tmp.iloc[x]
        row2 = tmp.iloc[y]

        # IGNORE MUTATIONS WITH DIFFERENT PH
        if row1.pH != row2.pH: continue
        # IGNORE INSERT AND DELETE MUTATIONS
        if row1['len'] != row2['len']: continue
        dd = {}

        dd['seq1'] = row1["protein_sequence"]
        dd['tm1'] = row1["tm"]
        dd['ds1'] = row1["data_source"]
        dd['seq2'] = row2["protein_sequence"]
        dd['tm2'] = row2["tm"]
        dd['ds2'] = row2["data_source"]
        dd['pH'] = row1['pH']
        dd['len'] = row1['len']
        dd['group'] = row1['group']
        rows.append(dd)

print()
print('All pairs of train single point edit mutations:')
df = pd.DataFrame(rows)
print('Dataframe shape', df.shape)
df.head()


def get_mutation(row):
    for i, (a, b) in enumerate(zip(row.seq1, row.seq2)):
        if a != b: break
    row['AA1'] = row.seq1[i]
    row['AA2'] = row.seq2[i]
    row['position'] = i
    return row


df = df.apply(get_mutation, axis=1)
df.to_csv(f'train_single_edit_mutations_v{VER}.csv', index=False)
df.head()