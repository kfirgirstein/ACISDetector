r"""
Use this module to write hyper parameters in the notebook.

"""


def random_forest_hp():
    estimators,random_state,n_jobs = 100 , 0 ,-1
    return dict(estimators=estimators, random_state=random_state, n_jobs=n_jobs)

def mlp_hp():
    lr, reg = 0.01, 0.05
    hidden_size = [100]
    return dict(hidden_size=hidden_size, lr=lr, reg=reg)

def cnn_hp():
    wstd, lr, = 0.1, 0.001
    return dict(wstd=wstd, lr=lr)