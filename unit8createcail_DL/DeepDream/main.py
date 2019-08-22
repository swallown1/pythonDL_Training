from keras.applications import inception_v3
from keras import backend as K
import scipy
import numpy as np
from keras.preprocessing import image


class  DeepDream():
    def __init__(self,image_path):
        self.image_path = image_path
        self.save_fname = './data'
        #设置该模型的不同损失的各自损失贡献
        self.layers_contributions = {
            'mixed2':0.2,
            'mixed3': 3.,
            'mixed4': 2.,
            'mixed5': 1.5,
        }
    # 将一个图片转化成一个可以放进inception V3的张量
    def preprocess_image(self):
        img = image.load_img(self.image_path)
        img = image.img_to_array(img)
        #表示在shape的第一维度加数据
        img = np.expand_dims(img,axis=0)
        img = inception_v3.preprocess_input(img)
        return img
    #将张量转化成图片
    def deprocess_image(self,x):
        if K.image_data_format() == 'channels_first':
            x =x.reshape((3,x.shape[2],x.shape[3]))
            x =x.transpose((1,2,0))
        else:
            x = x.reshape((x.shape[1],x.shape[2],3))
        x /=2.
        x +=0.5
        x *=255.
        x = np.clip(x,0,255).astype('uint8')
        return x
    #保存图片
    def save_img(self,img):
        pil_img = deprocess_image(np.copy(img))
        scipy.misc.imsave(self.save_fname,pil_img)

    def resize_img(self,img,size):
        img = img.copy(img)
        factors = (1,float(size[0])/img.shape[1],float(size[1])/img.shape[2],1)
        return scipy.ndimage.zoom(img,factors,order=1)

    def get_inception_V3(self):
        #禁止这个模型的训练操作
        K.set_learning_phase(0)
        model = inception_v3.InceptionV3(weight='imagenet',include_top = False)
        return model