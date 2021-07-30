#--------------------------------------------------#
#       使用FCN做语义分割
#--------------------------------------------------#
import tensorflow as tf

import PIL.Image as Image

from fcn_main import FCN

'''
'''

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#----------- 设置面板：全局变量 ---------------------------#
fcn = FCN()
detect_image = fcn.detect_image
#--------------------------------------------------------#

# 方式1：对单张图片做语义分割
while True:
    img = input('Input image filename (to ESC, input "quit"): ')
    if img == 'quit':
        print('Programe ends.')
        break

    try:
        image = Image.open(img)
        image = image.convert('RGB')
    except:
        print('Open Error! Try again.')
        continue
    else:
        r_image = detect_image(image)
        r_image.show()