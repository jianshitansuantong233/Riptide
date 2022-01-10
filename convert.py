import tensorflow as tf
from riptide.binary.binary_funcs import *
from riptide.binary.binary_layers import Config
from riptide.get_models import get_model

actQ = DQuantize
weightQ = XQuantize
config = Config(actQ=actQ, weightQ=weightQ, bits=2.0)
with config:
   model = get_model('cnn_medium')
   dummy_in = tf.keras.layers.Input(shape=[32, 32, 3], batch_size=1)
   dummy_out = model(dummy_in)
   # model.load_weights('/root/models/cnn_medium_2A1W/model.ckpt-3910')
   import tvm
   from tvm import relay
   mod, params = relay.frontend.from_keras(model,shape={'input_1': [1, 32, 32, 3]}, layout='NHWC')
   target = tvm.garget.arm_cpu("rasp3b")
   with relay.build_config(opt_level=3):
      graph, lib, params = relay.build(mod, target=target, params=params)
