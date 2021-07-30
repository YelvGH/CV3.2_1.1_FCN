#---------------------------------------------------#
#       end-to-end 神经网络
#---------------------------------------------------#
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import PIL.Image as Image

'''
    # 网络建模注意：
        1.encoder部分，并非严格与FCN论文图的结构一致，但保持了：
            最终输出的heat_map.shap为(16,16,?)
            保持了32×下采样结构
            encoder使用了mobilenet迁移学习，因此对应的输入尺寸为(512,512,3)
        2.decoder部分，上采样使用的kernal_size, strides参数未知
        3.encoder部分，原论文途中，取到encoder的部分，都是池化后的直接结果
            本处用到下一次池化前的结果
'''
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor % 2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) /
                       factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def f(photo, num_classes, mode='32s'):
    assert mode in ['8s', '16s', '32s'], 'mode not supported: {}'.format(mode)

    # encoder
    m_net = keras.applications.mobilenet.MobileNet(include_top=False, 
                                                   weights='imagenet', 
                                                   input_shape=photo.shape[1:4])
    m_net_BlockendLayerIndex = [10,23,36,73,86] # blockIdx,multiple,layerIdx: 0,2×,10  1,4×,23  2,8×,36  3,16×,73  4,32×,86
    
    encoder_block_1 = keras.models.Sequential(m_net.layers[0:10])
    encoder_block_1.trainable = False
    o1 = encoder_block_1(photo)

    encoder_block_2 = keras.models.Sequential(m_net.layers[10:23])
    encoder_block_2.trainable = False
    encoder_block_2.build(input_shape=o1.shape)
    o2 = encoder_block_2(o1)
    
    encoder_block_3 = keras.models.Sequential(m_net.layers[23:36])
    encoder_block_3.trainable = False
    encoder_block_3.build(input_shape=o2.shape)
    o3 = encoder_block_3(o2)
    
    encoder_block_4 = keras.models.Sequential(m_net.layers[36:73])
    encoder_block_4.trainable = False
    encoder_block_4.build(input_shape=o3.shape)
    o4 = encoder_block_4(o3)

    encoder_block_5 = keras.models.Sequential(m_net.layers[73:])
    encoder_block_5.trainable = False
    encoder_block_5.build(input_shape=o4.shape)
    o5 = encoder_block_5(o4)

    # decoder + 预测网络
    if mode == '32s': # -> (None, 544,544,?)
        d5 = keras.layers.Conv2D(num_classes+1, (1,1), padding='same')(o5)
        k_i = keras.initializers.Constant(bilinear_upsample_weights(32, num_classes+1))
        o  = keras.layers.Conv2DTranspose(num_classes+1, 
                                          kernel_size=(64,64),
                                          strides=(32,32),
                                          padding='valid',
                                          activation='softmax',
                                          use_bias=False,
                                          kernel_initializer=k_i)(d5) # o.shape=(?,544,544,?)

    if mode == '16s':
        d5 = keras.layers.Conv2D(num_classes+1, (1,1), padding='same')(o5)
        k_i = keras.initializers.Constant(bilinear_upsample_weights(2, num_classes+1))
        d5  = keras.layers.Conv2DTranspose(num_classes+1, 
                                          kernel_size=(4,4),
                                          strides=(2,2),
                                          padding='valid',
                                        #   activation='softmax',
                                          use_bias=False,
                                          kernel_initializer=k_i)(d5) # d5.shape=(?,34,34,?)

        d4 = keras.layers.Conv2D(num_classes+1, (1,1), padding='same')(o4)
        d4 = tf.image.resize(d4, (34,34))

        m1 = d4 + d5

        k_i = keras.initializers.Constant(bilinear_upsample_weights(16, num_classes+1))
        o  = keras.layers.Conv2DTranspose(num_classes+1, 
                                          kernel_size=(32,32),
                                          strides=(16,16),
                                          padding='valid',
                                          activation='softmax',
                                          use_bias=False,
                                          kernel_initializer=k_i)(m1) # o.shape=(?,?,?,?)

    if mode == '8s':
        d5 = keras.layers.Conv2D(num_classes+1, (1,1), padding='same')(o5)
        k_i = keras.initializers.Constant(bilinear_upsample_weights(2, num_classes+1))
        d5  = keras.layers.Conv2DTranspose(num_classes+1, 
                                          kernel_size=(4,4),
                                          strides=(2,2),
                                          padding='valid',
                                        #   activation='softmax',
                                          use_bias=False,
                                          kernel_initializer=k_i)(d5) # d5.shape=(?,34,34,?)

        d4 = keras.layers.Conv2D(num_classes+1, (1,1), padding='same')(o4)
        d4 = tf.image.resize(d4, (34,34))

        m1 = d4 + d5

        k_i = keras.initializers.Constant(bilinear_upsample_weights(2, num_classes+1))
        m1  = keras.layers.Conv2DTranspose(num_classes+1, 
                                          kernel_size=(4,4),
                                          strides=(2,2),
                                          padding='valid',
                                        #   activation='softmax',
                                          use_bias=False,
                                          kernel_initializer=k_i)(m1) # o.shape=(?,70,70,?)

        d3 = keras.layers.Conv2D(num_classes+1, (1,1), padding='same')(o3)
        d3 = tf.image.resize(d3, (70,70))
                 
        m2 = d3 + m1

        k_i = keras.initializers.Constant(bilinear_upsample_weights(8, num_classes+1))
        o  = keras.layers.Conv2DTranspose(num_classes+1,
                                          kernel_size=(16,16),
                                          strides=(8,8),
                                          padding='valid',
                                          activation='softmax',
                                          use_bias=False,
                                          kernel_initializer=k_i)(m2) # o.shape=(?,568,568,?)
 
    return o

def get_net_model(net_input_shape, num_classes, num_upSample=3, encoder_level=3, mode='32s'):
    assert net_input_shape == (512,512,3), 'net_input_shape should only be (512,512,3), but {} is given.'.format(net_input_shape)
    # 1.输入层
    i_put = keras.layers.Input(shape=net_input_shape)

    # 2.主干特征提取网络 + 加强特征提取网络
    decoded_map = f(i_put, num_classes, mode=mode)

    # 4.输出层
    o_put = tf.image.resize(decoded_map, net_input_shape[0:2]) # 还原到网络输入尺寸！
    # o_put = keras.layers.Softmax()(o_put)

    # 5.模型
    net_model = keras.models.Model(i_put, o_put)

    return net_model

if __name__ == '__main__':
    print('TST MSG: 测试FCN网络建模是否正确')
    net_model = get_net_model((512,512,3), 5, mode='8s')
    net_model.summary()
