#-------------------------------------------#
#       程序主体：前处理 + 网络 + 后处理
#-------------------------------------------#
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np 
import PIL.Image as Image

import nets.fcn_net as fcn_net

'''
    # 参考案例：
        
    # 网络建模注意：
        1.
'''

class FCN(object):
    _defaults = {
        'net_input_shape'     : (512,512,3), # 为了凑出32× feature_map的尺寸为(16,16,???)
        'net_preprocess_input': keras.applications.mobilenet.preprocess_input, 

        'get_net_model'       : fcn_net.get_net_model, 
        'fcn_mode'            : '8s',
        'load_classname_list' : False,
        'class_names'         : [],
        'classes_path'        : './model_data/voc_classes.txt',
        'num_classes'         : 3, # 语义分割数据集的物体类别数量  P.S. 不含背景类别！
        'load_weights'        : False,
        'model_weights_path'  : './model_data/w.h5',

        'blend_percent'       : 0.3, # 生成的混合图（原图与分割图的混合）时，分割图的混合比例
    }

    @classmethod
    def get_class_attribute(cls, key):
        if key in cls._defaults:
            return cls._defaults[key]
        else:
            return 'Unrecognized class attribute name: "{}" '.format(key)

    def __init__(self):
        self.__dict__.update(self._defaults)
        if self.load_classname_list:
            self.class_names = self._get_class_names()
            self.num_classes = len(self.class_names)
        self.generate_instance_attributes()

    def _get_class_names(self):
        # classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate_instance_attributes(self):
        self.net_model = self.get_net_model(self.net_input_shape, self.num_classes, mode=self.fcn_mode)
        if self.load_weights:
            self.net_model.load_weights(self.model_weights_path)
            print('{} model loaded.'.format(self.model_weights_path))
        else:
            print('TEST MSG: no model_weights loaded!')

        # 仅使用了一个颜色通道(G通道 i.e.绿色蒙版)的分类着色方法
        self.classColors = []
        tmp_colors = [[0, int(255/(x+1)), 0] for x in range(self.num_classes)]
        self.classColors.extend(tmp_colors)

    @tf.function(experimental_relax_shapes=True)
    def get_net_pred(self, photo):
        preds = self.net_model(photo, training=False)
        return preds

    def detect_image(self, image):
        ###----- 1.前处理
        resized_image = image.resize(self.net_input_shape[0:2], Image.BICUBIC)

        photo = np.asarray(resized_image)
        photo = np.expand_dims(photo, axis=0)
        photo = self.net_preprocess_input(photo)

        ###----- 2.网络
        pred = self.get_net_pred(photo)
        pred = pred[0].numpy()

        cond_1 = pred < 0
        cond_2 = pred > 1
        assert not(cond_1.any() or cond_2.any()), 'There is illegal value in pred, which is not in [0,1]'

        ###----- 3.后处理
        pred_classIndex = pred.argmax(axis=-1)

        seg_image = np.zeros((pred.shape[0], pred.shape[1], 3))
        for c in range(self.num_classes):
            seg_image[:,:,0] += (pred_classIndex==c) * self.classColors[c][0]
            seg_image[:,:,1] += (pred_classIndex==c) * self.classColors[c][1]
            seg_image[:,:,2] += (pred_classIndex==c) * self.classColors[c][2]
        seg_image = Image.fromarray(np.uint8(seg_image))
        seg_image = seg_image.resize((image.width, image.height))

        r_image = Image.blend(image, seg_image, alpha=self.blend_percent)


        return r_image