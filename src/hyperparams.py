r"""
Use this module to write hyper parameters in the notebook.

"""


def random_forest_hp():
    wstd, lr, reg = 0.1, 0.01, 0.05
    return dict(wstd=wstd, lr=lr, reg=reg)

def mlp_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, =  0.1, 0.01, 0.02, 0.00001, 0.01
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)

def cnn_hp():
    wstd, lr, = 0.1, 0.001
    return dict(wstd=wstd, lr=lr)