def set_random_seed(seed):

    from os import environ
    environ["PYTHONHASHSEED"] = '0'
    environ["CUDA_VISIBLE_DEVICES"]='-1'
    environ["TF_CUDNN_USE_AUTOTUNE"] ='0'

    import random
    random.seed(seed)

    from numpy.random import seed as np_seed
    np_seed(seed)

    from tensorflow import set_random_seed as tf_seed
    tf_seed(seed)
