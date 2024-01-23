import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, AveragePooling3D, Flatten, Activation, Dense
from tensorflow.keras.models import Model
from i3d_inception import Inception_Inflated3d
from unet_model_ import unet_model

# Parse command line arguments
epoch = int(sys.argv[1])
global_num_clients = int(sys.argv[2])
variation_idx = int(sys.argv[3])


temp = [13.0,55,92,129,24,96,118,328,241,150,26,14,78,19,3,28]
clients_scaling_factor = np.array(temp)
clients_scaling_factor = clients_scaling_factor[0:global_num_clients]
tot_size = np.sum(clients_scaling_factor)
clients_scaling_factor = clients_scaling_factor/tot_size


def main(_):
    global_model = unet_model()
    models = []
    for i in range(global_num_clients):
        filename = f"checkpoints_3d_variation{variation_idx}/client{i}epochend{epoch}.h5"
        model = unet_model()
        model.load_weights(filename)
        models.append(model)

    # Aggregate weights from different models
    weights = [model.get_weights() for model in models]
    new_weights = [np.average(np.array(w), axis=0, 
                              weights=clients_scaling_factor) 
                              for w in zip(*weights)]

    # Update global model with new weights
    global_model.set_weights(new_weights)
    fl_ckpt_path = f"checkpoints_3d_variation{variation_idx}/epoch{epoch}global.h5"
    
    # A version of FL combines the optimizer parameters and states as well. 
    # For FLAvg, this is ignored.
    global_model.save(fl_ckpt_path, 
                      include_optimizer=False)

if __name__ == '__main__':
    tf.compat.v1.app.run(main)
