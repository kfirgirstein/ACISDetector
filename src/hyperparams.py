r"""
Use this module to write hyper parameters in the notebook.

"""


def random_forest_hp():
    estimators,random_state,n_jobs = 100 , 0 ,-1
    return dict(estimators=estimators, random_state=random_state, n_jobs=n_jobs)

def mlp_hp():
    lr,reg = 0.01,0.05
    hidden_size = [100]
    return dict(hidden_size=hidden_size, lr=lr, reg=reg)

def mlp_hp_raw():
    lr,momentum,dropout = 0.001,0.9,0.0
    hidden_size = [300]
    return dict(hidden_size=hidden_size, lr=lr, momentum=momentum,dropout=dropout)

def cnn_hp():
    lr,k,s,p,d= 0.0001,8,6,0,1
    h_c = [10, 100, 200]
    return dict(lr=lr,k=k,s=s,h_c=h_c,p=p,d=d)

def rnn_hp():
    i_s, l, h_f, lr, = 50, 2, 512, 0.0001
    return dict(i_s=i_s, l=l, h_f=h_f, lr=lr)