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
    lr,momentum,dropout = 0.00075,0.9,0.0
    hidden_size = [300]
    #hidden_size = [750,300]
    return dict(hidden_size=hidden_size, lr=lr, momentum=momentum,dropout=dropout)

def cnn_hp():
    lr,k,s,p,d= 0.001,4,2,0,1
    h_c = [1,1,1,1]
    return dict(lr=lr,k=k,s=s,h_c=h_c,p=p,d=d)

def rnn_hp():
    l,h_f, lr, =2,512, 0.001
    return dict(l=l,h_f = h_f, lr=lr)