#---------------------------------------#
#       train 工具函数、工具类
#---------------------------------------#
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import PIL.Image as Image

'''
    # 内容：
        1. metrics fucntions
        2. loss functions
        3. Generator
'''
###----- 1. metrics functions
def Iou_score(smooth = 1e-5, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = keras.backend.greater(y_pred, threhold)
        y_pred = keras.backend.cast(y_pred, keras.backend.floatx())
        
        intersection = keras.backend.sum(y_true * y_pred, axis=[0,1,2])
        union = keras.backend.sum(y_true + y_pred, axis=[0,1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score
    return _Iou_score

def f_score(beta=1, smooth = 1e-5, threhold = 0.5):
    def _f_score(y_true, y_pred):
        y_pred = keras.backend.greater(y_pred, threhold)
        y_pred = keras.backend.cast(y_pred, keras.backend.floatx())

        tp = keras.backend.sum(y_true * y_pred, axis=[0,1,2])
        fp = keras.backend.sum(y_pred, axis=[0,1,2]) - tp
        fn = keras.backend.sum(y_true, axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score
    return _f_score


###----- 2. loss functions
def get_loss_func(coef=1.0): 
    def _loss(gt, pred):
        loss = keras.backend.categorical_crossentropy(gt, pred) # 这个函数的输入，一定是严格的独热向量 & softmax结果。否则，计算结果准不准确不了解
        loss = keras.backend.mean(loss) * coef
        return loss
    return _loss


###----- 3. Generator
class SegmentationClass_Generator(object):
    def __init__(self, annotation_path,
                       multi_class_mode,
                       batch_size,
                       net_input_shape,
                       net_preprocess_input,
                       num_classes,
                       net_output_shape):
        # self.annotation_path        = annotation_path
        self.multi_class_mode       = multi_class_mode
        self.batch_size             = batch_size
        self.net_input_shape        = net_input_shape
        self.net_preprocess_input   = net_preprocess_input
        self.num_classes            = num_classes
        self.net_output_shape       = net_output_shape

        if not self.multi_class_mode: 
            assert self.num_classes == 1, 'in foreground-background train mode, num_classes should be 1, but num_classes={} was given.'.format(self.num_classes)

        # Part I: 读取数据记录文本，获得生样本信息列表
        with open(annotation_path) as f:
            self.raw_samples = f.readlines()
        np.random.seed(seed=1000)
        np.random.shuffle(self.raw_samples)
        np.random.seed(seed=None)

    def generate(self):
        count_i = 0
        photo_batch  = []
        onehot_batch = []

        while True:
            np.random.shuffle(self.raw_samples)
            
            for line in self.raw_samples:
                # 2.1 解析一条生样本，获得图片路径列表、边界框信息
                line_parse = line.split()[0]  # 
                # # TST
                # print(line_parse)
                image_dir = 'D:/【AI】/Datasets/CV_ds/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/' + line_parse + '.jpg'
                label_dir = 'D:/【AI】/Datasets/CV_ds/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/SegmentationClass/' + line_parse + '.png'
                image = Image.open(image_dir)
                label = Image.open(label_dir)

                # 2.2 image -> photo
                resized_image = image.resize(self.net_input_shape[0:2], Image.BICUBIC)

                photo = np.asarray(resized_image)
                # photo = np.expand_dims(photo, axis=0)
                photo = self.net_preprocess_input(photo)

                # 2.3 label -> one-hot
                resized_label = label.resize(self.net_output_shape[0:2], Image.NEAREST) # 对于label，要用近邻插值

                png = np.array(resized_label).astype(np.int32) 
                png[png == 255] = 0
                png = png - 1
                if not self.multi_class_mode: # 前景-背景训练模式，把所有物体类别的标记都修改为前景标记0
                    png[png >= 0] = 0
                png[png == -1] = self.num_classes
                # # TST:
                # print(np.unique(png))

                onehot_label = np.eye(self.num_classes+1)[png.reshape([-1])]  # 独热向量还可以用xx_main.py中生成seg_image的方式生成
                onehot_label = onehot_label.reshape([self.net_output_shape[0],
                                                     self.net_output_shape[1],
                                                     self.num_classes+1])

                # 2.4 yield
                photo_batch.append(photo)
                onehot_batch.append(onehot_label)
                count_i += 1
                if count_i >= self.batch_size:
                    tmp_1 = np.array(photo_batch).astype(np.float32)  # shape=(batch_size,h,w,3)
                    tmp_2 = np.array(onehot_batch).astype(np.float32) # shape=(batch_size,h,w,num_classes) 虽然是独热向量，但是为了给tensor计算，最好用np.float32
                    count_i = 0
                    photo_batch = []
                    onehot_batch = []

                    yield tmp_1, tmp_2

if __name__ == '__main__':
    # TST: 测试类生成器
    import matplotlib.pyplot as plt
    gen = SegmentationClass_Generator(annotation_path = './model_data/train.txt',
                                      multi_class_mode = True,
                                      batch_size = 8,
                                      net_input_shape = (416,416,3),
                                      net_preprocess_input = keras.applications.mobilenet.preprocess_input,
                                      num_classes = 20,
                                      net_output_shape = (208,208)).generate
    for i, (photo, label) in enumerate(gen()):
        for j in range(8): # batch_size
            # 显示图像
            tmp = (photo[j]*128 + 128).astype(np.uint8)
            tmp = Image.fromarray(tmp)
            tmp.show()
            # 显示标签
            for k in range(20): # num_classes
                if label[j][:,:,k].max() == 1:
                    print('k = {}'.format(k))
                    tmp = label[j][:,:,k]*255
                    tmp = Image.fromarray(tmp) # 255
                    tmp.show()
        if i == 2:
            break
            