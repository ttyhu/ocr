import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.recurrent import GRU
from keras.layers.core import Reshape, Dense, Flatten, Permute, Lambda, Activation
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras import regularizers
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


def rpn_loss_regr(y_true, y_pred):
    """
    smooth L1 loss

    y_ture [1][HXWX9][3] (class,regr)
    y_pred [1][HXWX9][2] (reger)
    """

    sigma = 9.0

    cls = y_true[0, :, 0]
    regr = y_true[0, :, 1:3]
    regr_keep = tf.where(K.equal(cls, 1))[:, 0]
    regr_true = tf.gather(regr, regr_keep)
    regr_pred = tf.gather(y_pred[0], regr_keep)
    diff = tf.abs(regr_true - regr_pred)
    less_one = tf.cast(tf.less(diff, 1.0 / sigma), 'float32')
    loss = less_one * 0.5 * diff ** 2 * sigma + tf.abs(1 - less_one) * (diff - 0.5 / sigma)
    loss = K.sum(loss, axis=1)

    return K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.0))


def rpn_loss_cls(y_true, y_pred):
    """
    softmax loss

    y_true [1][1][HXWX9] class
    y_pred [1][HXWX9][2] class
    """
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
    cls_true = tf.gather(y_true, cls_keep)
    cls_pred = tf.gather(y_pred[0], cls_keep)
    cls_true = tf.cast(cls_true, 'int64')
    # loss = K.sparse_categorical_crossentropy(cls_true,cls_pred,from_logits=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
    return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


def nn_base(input, trainable):
    base_model = VGG16(weights=None, include_top=False, input_shape=input)
    base_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if (trainable == False):
        for ly in base_model.layers:
            ly.trainable = False
    return base_model.input, base_model.get_layer('block5_conv3').output


def reshape(x):
    b = tf.shape(x)
    x = tf.reshape(x, [b[0] * b[1], b[2], b[3]])
    return x


def reshape2(x):
    x1, x2 = x
    b = tf.shape(x2)
    x = tf.reshape(x1, [b[0], b[1], b[2], 256])
    return x


def reshape3(x):
    b = tf.shape(x)
    x = tf.reshape(x, [b[0], b[1] * b[2] * 10, 2])
    return x


def rpn(base_layers):
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
               name='rpn_conv1')(base_layers)

    x1 = Lambda(reshape, output_shape=(None, 512))(x)

    x2 = Bidirectional(GRU(128, return_sequences=True), name='blstm')(x1)

    x3 = Lambda(reshape2, output_shape=(None, None, 256))([x2, x])
    x3 = Conv2D(512, (1, 1), padding='same', activation='relu', name='lstm_fc')(x3)

    cls = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_class')(x3)
    regr = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_regress')(x3)

    cls = Lambda(reshape3, output_shape=(None, 2), name='rpn_class_reshape')(cls)

    regr = Lambda(reshape3, output_shape=(None, 2), name='rpn_regress_reshape')(regr)

    return cls, regr


inp, nn = nn_base((None, None, 3), trainable=True)
cls, regr = rpn(nn)
basemodel = Model(inp, [cls, regr])
basemodel.summary()

import utils

xmlpath = 'VOCdevkit/Annotations'
imgpath = 'VOCdevkit/JPEGImages'
gen1 = utils.gen_sample(xmlpath, imgpath, 1)
gen2 = utils.gen_sample(xmlpath, imgpath, 1)


class losslog():
    def __init__(self, path, txt):
        with open(path, 'a+') as f:
            f.writelines(txt)


class losshistroy(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        txtloss = str(logs.get('rpn_class_reshape_loss')) + ',' + str(logs.get('rpn_regress_reshape_loss')) + '\r\n'
        losslog('loss3.cvs', txtloss)
        # print('rpn_class_reshape_loss:',logs.get('rpn_class_reshape_loss'),' ','rpn_regress_reshape_loss:',logs.get('rpn_regress_reshape_loss'))


hisloss = losshistroy()

checkpoint = ModelCheckpoint("model/weights.{epoch:03d}-{loss:.3f}.hdf5",
                             monitor="loss",
                             verbose=1,
                             mode="auto",
                             save_best_only=True)
earlystop = EarlyStopping(patience=10)

utils.get_session(gpu_fraction=0.6)

# sgd = SGD(0.0001,0.9,nesterov = True)
adam = Adam(0.00001)
# adam = RMSprop(0.00001)
basemodel.compile(optimizer=adam,
                  loss={'rpn_class_reshape': rpn_loss_cls, 'rpn_regress_reshape': rpn_loss_regr},
                  loss_weights={'rpn_class_reshape': 1.0, 'rpn_regress_reshape': 1.0}
                  )

res = basemodel.fit_generator(gen1, 6000, epochs=1000, verbose=1, callbacks=[checkpoint, hisloss])
