def set_seed(seed):
    
    from os import environ
    environ["PYTHONHASHSEED"] = '0'
    environ["CUDA_VISIBLE_DEVICES"]='-1'
    environ["TF_CUDNN_USE_AUTOTUNE"] ='0'

    import tensorflow as tf

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
     
    from numpy.random import seed as np_seed
    np_seed(seed)
    import random
    random.seed(seed)

    tf.set_random_seed(seed)
