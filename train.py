#-----------------------------------------------------#
#       训练框架：准备模型>>准备数据>>训练模型，@VOC
#-----------------------------------------------------#
import tensorflow as tf
import tensorflow.keras as keras

import PIL.Image as Image
import time
import os
import numpy as np

from fcn_main import FCN as XX_NET_proto
import utils.utils_train as utils_train

'''
    # 训练框架实现：
        1.读入模型、读入训练数据，配置训练参数，训练
        2.保存训练参数配置、训练记录
'''
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#------- 设置面板：全局变量 ------------------------------------#
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)          # 屏蔽tensorflow的warning

# tf.config.threading.set_inter_op_parallelism_threads(2) # 控制计算使用的线程 i.e. CPU资源
# tf.config.threading.set_intra_op_parallelism_threads(2)

# 1.准备模型
class XX_NET(XX_NET_proto):
    def __init__(self):
        super()._defaults['num_classes']         = 1                    # 修改类属性：训练是否需要加载分类名称表
        super()._defaults['fcn_mode']            = '32s'
        super()._defaults['load_classname_list'] = False
        super()._defaults['load_weights']        = False                # 修改类属性：训练是否需要加载权重
        super()._defaults['model_weights_path']  = './logs/proto_w.h5'
        super().__init__()                                              # 实例化方法
xx_net = XX_NET()

# 2.准备数据
annotation_path_train   = './model_data/train.txt'
annotation_path_valid   = './model_data/val.txt'

Gen_CLASS = utils_train.SegmentationClass_Generator

# 3.准备训练参数
class TRAIN_PARA(object):
    _defaults = {
        'initial_lr'                : 1e-3,

        'reduce_lr_factor'          : 0.3,
        'reduce_lr_patience'        : 3,
        'early_stopping_min_delta'  : 0,
        'early_stopping_patience'   : 7,

        'log_dir'                   : 'to_be_updated',

        'batch_size': 4,
        'epoches'         : 50,
        'steps_per_epoch' : 10,
        'validation_steps': 10,
        'validation_freq' : 1,

        'notebook': '--\n\n' + 'Training-descriptions: \n' \
                    + '  ' + '__contents__' + '\n' \
                    + '  ' + '__contents__',
    }
    def __init__(self):
        self.__dict__.update(self._defaults)
        # 建立文件目录
        train_time = time.localtime()
        self.log_dir    = './logs/{}-{:0>2d}-{:0>2d}_{:0>2d}{:2>2d} lr={:.2e}/'.format(train_time.tm_year, 
                                                                                  train_time.tm_mon,
                                                                                  train_time.tm_mday, 
                                                                                  train_time.tm_hour,
                                                                                  train_time.tm_min,
                                                                                  self.initial_lr)
        if os.path.exists(self.log_dir):
            print('dir "{}" exists!'.format(self.log_dir))
        else:
            os.mkdir(self.log_dir)
        # 保存训练参数
        with open(file=self.log_dir + '_train_para_configurations.txt', mode='x') as f:
            for key in self.__dict__:
                f.writelines('{:26}: {}\n'.format(key, self.__dict__[key]))
train_para = TRAIN_PARA()
#-------------------------------------------------------------#


###----- 1.准备模型
net_model = xx_net.net_model
net_model.summary()


###----- 2.准备数据
tmp = net_model(np.expand_dims(np.random.random(xx_net.net_input_shape), axis=0)).numpy().shape
train_gen = Gen_CLASS(annotation_path   =annotation_path_train, 
                      multi_class_mode  =xx_net.load_classname_list,
                      batch_size        = train_para.batch_size,
                      net_input_shape   = xx_net.net_input_shape,
                      net_preprocess_input =xx_net.net_preprocess_input,
                      num_classes       = xx_net.num_classes,
                      net_output_shape  = tmp[1:3]).generate()
valid_gen = Gen_CLASS(annotation_path   =annotation_path_valid, 
                      multi_class_mode  =xx_net.load_classname_list,
                      batch_size        = train_para.batch_size,
                      net_input_shape   = xx_net.net_input_shape,
                      net_preprocess_input =xx_net.net_preprocess_input,
                      num_classes       = xx_net.num_classes,
                      net_output_shape  = tmp[1:3]).generate()


###----- 3.训练模型
# 3.1 .compile()
loss_func = utils_train.get_loss_func() # 'categorical_crossentropy' # 
optm_func = keras.optimizers.Adam(lr=train_para.initial_lr)
metr_list = [utils_train.f_score(), utils_train.Iou_score()]

net_model.compile(loss=loss_func,
                  optimizer=optm_func, # ) # 
                  metrics=metr_list) 

# 3.2 .fit()
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath = train_para.log_dir + 'weights.epoch_{epoch:02d}-loss_{loss:.2f}-val_loss_{val_loss:.2f}.h5',
    monitor            ='loss', # 'val_loss' # 
    save_weights_only  =True, 
    save_best_only     =False,
    save_freq          ='epoch')
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor   ='loss', # 'val_loss' # 
    factor    =train_para.reduce_lr_factor,
    patience  =train_para.reduce_lr_patience, 
    verbose   =1)
early_stopping = keras.callbacks.EarlyStopping(
    monitor      ='loss', # 'val_loss' # 
    min_delta    =train_para.early_stopping_min_delta, 
    patience     =train_para.early_stopping_patience, 
    verbose      =1)
tensorboard = keras.callbacks.TensorBoard(log_dir=train_para.log_dir)
callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard] # ] # 

train_history = net_model.fit(x                 =train_gen,  
                              epochs            =train_para.epoches, 
                              steps_per_epoch   =train_para.steps_per_epoch, 
                              callbacks         =callbacks,
                              validation_data   =valid_gen,
                              validation_steps  =train_para.validation_steps,
                              validation_freq   =train_para.validation_freq)
net_model.save_weights(train_para.log_dir+'weights.h5')

'''
    # Tensorboard：keras中的使用方法
        1.在callbacks中添加，并设置保存目录
        2.在保存目录中启动cmd，输入如下命令
            tensorboard --logdir ./
        3.在浏览器中打开网址 http://localhost:6006/
'''