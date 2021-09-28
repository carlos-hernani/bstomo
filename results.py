# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Results: Quantum Pattern Recognition with Boson-Sampling

# %%
import os
import numpy as np
from utils import *

# Loading the data
file_paths = os.path.expanduser("~") + '/Dropbox/BS-Tomography/RegresionData/Data4Modes/4Modes_'
file_names = (
    '1photon/Data2.npz',
    '2photon/Data_2photons.npz',
    '3photons/Data_3photons.npz',
    'entanglement/entanglement_state_1/Entanglement_state.npz',
    'entanglement/entanglement_state_2/Entanglement_state_2photons.npz',
    'entanglement/entanglement_state_3/Entanglement_state_3photons.npz'
    )
names = (
    '1photon',
    '2photon',
    '3photon',
    'ent1',
    'ent2',
    'ent3'
)

# NOTE: a for angles
param_names = dict(zip(names ,[
    ['a.0', 'a.1'],
    ['r.0', 'r.1', 'r.2', 'a.0', 'a.1', 'a.2'],
    ['r.0', 'r.1', 'r.2', 'r.3', 'a.0', 'a.1', 'a.2', 'a.3'],
    ['E'],
    ['E'],
    ['E']
]))

data = {name: load_data(file_paths+path) for name, path in zip(names, file_names)}

X = {}
P = {}
for name in names:
    X[name], P[name] = get_dataframe(data[name], param_names[name])

# Dimensionality Reduction using PCA
X_PCA = {name: pca(X[name]) for name in names}
print(f'Before and after dimensionality reduction explained variance ratio at {pca.__defaults__[0]}\n')
_ = [print(f'{name} : {X[name].shape[1]} \t {X_PCA[name].shape[1]}') for name in names]

# Training and Test Split
xy = {name: tts(X[name], P[name], test_size=0.2, random_state=0) for name in names}
xy_pca = {name: tts(X_PCA[name], P[name], test_size=0.2, random_state=0) for name in names}

results_fidelity = []
results_fidelity_pca = []

# %% [markdown]
# ## Description of the data
# 
# We have 6 datasets:
# 
# * 1 photon : two parameters ($\beta, \phi_1$)
# * 2 photon : five parameters ($r_0, r_1, r_2$), ($\phi_0, \phi_1, \phi_2$); $\phi_0 = 0$
# * 3 photon : seven parameters ($r_0, r_1, r_2, r_3$), ($\phi_0, \phi_1, \phi_2, \phi_3$); $\phi_0 = 0$
# * Entangled 1: one parameter $E$
# * Entangled 2: one parameter $E$
# * Entangled 3: one parameter $E$
# %% [markdown]
# ### 4 MODES 1 PHOTON
# 
# $|\eta_3 > = cos \beta |0> + sin \beta e^{i\phi_1} = r_0|0> + r_1 e^{i\phi_1}$
# 

# %%
name = '1photon'
P[name].head()


# %%
model = etr_reg(xy[name])
scores(model)
model_pca = etr_reg(xy_pca[name])
scores(model)


# %%
# pca and non-pca give the same R2-score so we use the simplest one
fid1_etr = fidelity_1(real=model['real'] ,predictions=model['preds'])
res1_etr = pd.DataFrame(fid1_etr, columns=['1_etr'])
results_fidelity.append(res1_etr)
res1_etr.describe()


# %%
# pca and non-pca give the same R2-score so we use the simplest one
fid1_etr = fidelity_1(real=model_pca['real'] ,predictions=model_pca['preds'])
res1_etr = pd.DataFrame(fid1_etr, columns=['1_etr'])
results_fidelity_pca.append(res1_etr)
res1_etr.describe()


# %%
model = svr_reg(xy[name])
scores(model)
model_pca = svr_reg(xy_pca[name])
scores(model_pca)


# %%
# pca and non-pca give almost the same R2-score so we use the simplest one
fid1_svr = fidelity_1(real=model['real'] ,predictions=model['preds'])
res1_svr = pd.DataFrame(fid1_svr, columns=['1_svr'])
results_fidelity.append(res1_svr)
res1_svr.describe()


# %%
# pca and non-pca give almost the same R2-score so we use the simplest one
fid1_svr = fidelity_1(real=model_pca['real'] ,predictions=model_pca['preds'])
res1_svr = pd.DataFrame(fid1_svr, columns=['1_svr'])
results_fidelity_pca.append(res1_svr)
res1_svr.describe()

# %% [markdown]
# ### 4 MODES 2 PHOTON

# %%
name = '2photon'
P[name].head()


# %%
model = etr_reg(xy[name])
scores(model)
model_pca = etr_reg(xy_pca[name])
scores(model)


# %%
npreds = norm(model['preds'])
npreds.head()
fid2_etr = fidelity_2(model['real'], npreds)
res2_etr = pd.DataFrame(fid2_etr, columns=['2_etr'])
results_fidelity.append(res2_etr)
res2_etr.describe()


# %%
npreds = norm(model_pca['preds'])
npreds.head()
fid2_etr = fidelity_2(model_pca['real'], npreds)
res2_etr = pd.DataFrame(fid2_etr, columns=['2_etr'])
results_fidelity_pca.append(res2_etr)
res2_etr.describe()


# %%
model = svr_reg(xy[name])
scores(model)
model_pca = svr_reg(xy_pca[name])
scores(model_pca)


# %%
npreds = norm(model['preds'])
npreds.head()
fid2_svr = fidelity_2(model['real'], npreds)
res2_svr = pd.DataFrame(fid2_svr, columns=['2_etr'])
results_fidelity.append(res2_svr)
res2_svr.describe()


# %%
npreds = norm(model_pca['preds'])
npreds.head()
fid2_svr = fidelity_2(model_pca['real'], npreds)
res2_svr = pd.DataFrame(fid2_svr, columns=['2_etr'])
results_fidelity_pca.append(res2_svr)
res2_svr.describe()

# %% [markdown]
# ### 4 MODES 3 PHOTON

# %%
name = '3photon'
P[name].head()


# %%
model = etr_reg(xy[name])
scores(model)
model_pca = etr_reg(xy_pca[name])
scores(model)


# %%
npreds = norm(model['preds'])
npreds.head()
fid3_etr = fidelity_2(model['real'], npreds)
res3_etr = pd.DataFrame(fid3_etr, columns=['3_etr'])
results_fidelity.append(res3_etr)
res3_etr.describe()


# %%
npreds = norm(model_pca['preds'])
npreds.head()
fid3_etr = fidelity_2(model_pca['real'], npreds)
res3_etr = pd.DataFrame(fid3_etr, columns=['3_etr'])
results_fidelity_pca.append(res3_etr)
res3_etr.describe()


# %%
model = svr_reg(xy[name])
scores(model)
model_pca = svr_reg(xy_pca[name])
scores(model_pca)


# %%
npreds = norm(model['preds'])
npreds.head()
fid3_svr = fidelity_2(model['real'], npreds)
res3_svr = pd.DataFrame(fid3_svr, columns=['3_svr'])
results_fidelity.append(res3_svr)
res3_svr.describe()


# %%
npreds = norm(model_pca['preds'])
npreds.head()
fid3_svr = fidelity_2(model_pca['real'], npreds)
res3_svr = pd.DataFrame(fid3_svr, columns=['3_svr'])
results_fidelity_pca.append(res3_svr)
res3_svr.describe()

# %% [markdown]
# ### Results for 4 MODES: 1,2,3 photons
# #### Without pca

# %%
pd.concat(results_fidelity, axis=1).describe()

# %% [markdown]
# ### Results for 4 MODES: 1,2,3 photons
# #### With pca

# %%
pd.concat(results_fidelity_pca, axis=1).describe()

# %% [markdown]
# ## 4 MODES ENTANGLEMENT
# 
# Here we don't have fidelity only the $R^2$-score to measure the performance of our model.
# %% [markdown]
# ### 4 MODES ENTANGLEMENT 1

# %%
name = 'ent1'
model = etr_reg(xy[name],oaa=True)
scores(model)
model_pca = etr_reg(xy_pca[name],oaa=True)
scores(model_pca)

# %% [markdown]
# ### 4 MODES ENTANGLEMENT 2

# %%
name = 'ent2'
model = etr_reg(xy[name],oaa=True)
scores(model)
model_pca = etr_reg(xy_pca[name],oaa=True)
scores(model_pca)

# %% [markdown]
# ### 4 MODES ENTANGLEMENT 3

# %%
name = 'ent3'
model = etr_reg(xy[name],oaa=True)
scores(model)
model_pca = etr_reg(xy_pca[name],oaa=True)
scores(model_pca)

# %% [markdown]
# # Summary
# 
# Firstly, I have to point that PCA reduces greatly the number of features needed to perform regression keeping almost the same or even better performance in some cases.
# 
# The 4 Modes for 1, 2 and 3 photons datasets have good results in terms of $R^2$-score for our regression algorithms (SVR and ExtraTreesRegressor) except for the first one. This prompts me to think that maybe the first dataset is quite different from the rest, because one can see a tendency of the scores to drop when introducing more photons, but in this case our predictions show a negative $R^2$-score which means that our regression is arbitrarily wrong, nonetheless fidelity calculation shows us that mean/median values (they are almost the same) are approximately 0.64 and 0.85 for etr & svr each.
# 
# In terms of fidelity, all the 3 datasets are quite good, with values improved with pca reduction.
# The $R^2$-scores for the last 3 datasets are also quite good, and also with values improved with pca reduction.

# %%



