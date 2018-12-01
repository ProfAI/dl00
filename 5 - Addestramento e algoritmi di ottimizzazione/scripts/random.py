def set_seed(seed):
    from os import environ
    environ["PYTHONHASHSEED"] = '0'
    environ["CUDA_VISIBLE_DEVICES"]='-1'
    environ["TF_CUDNN_USE_AUTOTUNE"] ='0'

    from numpy.random import seed as np_seed
    np_seed(seed)
    import random
    random.seed(seed)
    from tensorflow import set_random_seed
    set_random_seed(seed)
