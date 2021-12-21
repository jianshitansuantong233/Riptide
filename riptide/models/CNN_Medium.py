import tensorflow as tf
from riptide.binary import binary_layers as nn

bnmomemtum=0.9

class CNN_Medium(tf.keras.Model):
    def __init__(self):
        super(CNN_Medium, self).__init__()

        self.conv1 = nn.NormalConv2D(128, 3, padding='valid', activation='relu')
        self.bn1 = nn.NormalBatchNormalization(momentum=bnmomemtum)
        #self.mp0 = nn.MaxPool2D(pool_size=2)
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=2)
        self.conv2 = nn.BinaryConv2D(filters=128, kernel_size=3, activation='relu', padding='valid', use_bias=False)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.bn2 = nn.BatchNormalization(self.conv2, momentum=bnmomemtum)
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv3 = nn.BinaryConv2D(filters=256, kernel_size=3, activation='relu', padding='valid',  use_bias=False)
        self.bn3 = nn.BatchNormalization(self.conv3, momentum=bnmomemtum)
        self.pad3 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv4 = nn.BinaryConv2D(filters=256, kernel_size=3, activation='relu', padding='valid', use_bias=False)
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.bn4 = nn.BatchNormalization(self.conv4, momentum=bnmomemtum)
        self.pad4 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv5 = nn.BinaryConv2D(filters=512, kernel_size=3, activation='relu', padding='valid', use_bias=False)
        self.bn5 = nn.BatchNormalization(self.conv5, momentum=bnmomemtum)
        self.pad5 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv6 = nn.BinaryConv2D(filters=512, kernel_size=3, activation='relu', padding='valid', use_bias=False)
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.bn6 = nn.BatchNormalization(self.conv6, momentum=bnmomemtum)
        # Output
        self.conv7 = nn.BinaryConv2D(filters=10, kernel_size=4, activation='relu', padding='valid', use_bias=False)
        self.bn7 = nn.BatchNormalization(self.conv7, momentum=bnmomemtum)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        #y = self.mp0(y)
        x = self.pad1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)
        x = self.pad2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pad3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x, training=training)
        x = self.pad4(x)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.pad5(x)
        x = self.conv6(x)
        x = self.pool6(x)
        x = self.bn6(x, training=training)
        x = self.conv7(x)
        x = tf.reshape(x, [-1, 10])
        x = self.bn7(x)
        
        return self.softmax(x)
