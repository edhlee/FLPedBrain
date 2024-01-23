import sys
import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.layers import Input,  Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow import keras as K
# from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from i3d_inception import Inception_Inflated3d
from unet_model_ import unet_model



# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Precision policy setup
from tensorflow.keras import mixed_precision
use_fp16 = sys.argv[3].lower() == 'true'
policy_name = 'mixed_float16' if use_fp16 else 'float32'
policy = mixed_precision.Policy(policy_name)
mixed_precision.set_global_policy(policy)

# Print out policy details
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)




from tensorflow.keras import mixed_precision
use_fp16 = sys.argv[3].lower() == 'true'




# if (use_fp16):
#   policy = mixed_precision.Policy('mixed_float16')
#   policy = tf.keras.mixed_precision.set_global_policy('mixed_float16')
# else:
#   policy = mixed_precision.Policy('float32')
#   policy = tf.keras.mixed_precision.set_global_policy('float32')




use_priming = False
# reverse_train_val = sys.argv[1].lower() == 'true'
global_test_index = int(sys.argv[1])
global_offset = 50
np.random.seed(42)
global_offset_gold = int(sys.argv[4]) #50
threshold_gold = float(sys.argv[5]) #0.8
use_all_data = sys.argv[6].lower() == 'true'

global_starting_epoch = int(sys.argv[7])
global_client_idx = int(sys.argv[8])
global_num_clients = int(sys.argv[9])
global_num_local_epochs = int(sys.argv[10])
use_iid_data = sys.argv[11].lower() == 'true'
K=global_num_clients
client_frac = float(sys.argv[12])
num_active_clients = int(max(K*client_frac, 1.0))
chance_of_active_client = num_active_clients / K # 0.1*21 = 2/21~10%
use_client = np.random.binomial(1, chance_of_active_client, 1) > 0.5
dont_use_client = not use_client
variation_idx = int(sys.argv[13])
site_split_idx = int(sys.argv[14]) # 0 or 1
l2_norm_clip=float(sys.argv[15])
noise_multiplier=float(sys.argv[16])
num_microbatches=int(sys.argv[17])
use_only_synths= sys.argv[18].lower() == 'true'
#l2_norm_clip=20.0
#noise_multiplier=0.1
site_idx = global_client_idx
strength_of_dice_loss=0.5
np.random.seed(global_starting_epoch)
BATCH_SIZE = 8
BATCH_SIZE_EVAL = 2
mu=0.1


# normals + tumors:
# pickle_in = open('shards/num_shards_' + str(num_gpus) + '_idx_'+str(hvd.local_rank())+ '.npy', "rb")
# only abnormals
# pickle_in = open('shards/tumorOnly_num_shards_' + str(num_gpus) + '_idx_'+str(hvd.local_rank())+ '.npy', "rb")



@tf.function
def rotate_tf(image, c, roi):

    image = tf.cast(image, tf.float32)

    image /= 255.0

    # this is using the float16 format
    # image = image


#     label = tf.cast(label, tf.float32)
    c =  tf.cast(c, tf.float32)
    roi = tf.cast(roi,tf.float32)
#     image = (image/127.5) - 1

    flip = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0)
    flip_y = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=351)

    delta_x = tf.random.uniform(shape=[], maxval=16, dtype=tf.int32, seed=8)
    delta_y = tf.random.uniform(shape=[], maxval=16, dtype=tf.int32, seed=5)

    delta_time_start = tf.random.uniform(shape=[], maxval=17, dtype=tf.int32, seed=8)
    jump_frames_multiplier = tf.random.uniform(shape=[],minval=2, maxval=4, dtype=tf.int32, seed=0)

    r = tf.random.uniform(shape=[], maxval=11, dtype=tf.int32, seed=0)
    r2 = tf.random.uniform(shape=[], maxval=11, dtype=tf.int32, seed=3)
    resize_num = tf.random.uniform(shape=[], minval=120, maxval=210, dtype=tf.int32, seed=3)

    ys = []
    ys_mask = []
    label_of_plane_roi = []
    for i in range(64):
        image_2d = image[delta_x:delta_x+240, delta_y:delta_y+240, i, :]
#         label_of_plane_roi.append(label_of_plane[i+delta_time_start])
        image_mask_2d = roi[delta_x:delta_x+240, delta_y:delta_y+240, i, :]

        # if (flip == 1):
        #     image_2d = tf.image.flip_left_right(image_2d)
        #     image_mask_2d = tf.image.flip_left_right(image_mask_2d)
        # if (flip_y == 1):
        #     image_2d = tf.image.flip_up_down(image_2d)
        #     image_mask_2d = tf.image.flip_up_down(image_mask_2d)

        image_2d = tf.image.resize(image_2d, [256,256], preserve_aspect_ratio=False,antialias=False, name=None)
        image_mask_2d = tf.image.resize(image_mask_2d, [256,256], preserve_aspect_ratio=False,antialias=False, name=None)
#         image_mask_2d = tf.cast(image_mask_2d > 0.01, tf.float32)
        ys.append(image_2d)
        ys_mask.append(image_mask_2d)

    ys = tf.convert_to_tensor(ys, dtype=tf.float32)
#     label_of_plane_roi = tf.convert_to_tensor(label_of_plane_roi,  dtype=tf.float32)
    ys_mask = tf.convert_to_tensor(ys_mask, dtype=tf.float32)
#     ys = tf.transpose(ys, perm=[2,1,0,3])
    print("train shape:", ys.shape)
    print(ys)
    return ys,  c, ys_mask


# all_data = np.load("processed_trainval8020_March2020_256x256x24/uint8_2023_train886_val227_256x256x64.npy", allow_pickle=True)
institute = "institute_"+str(global_client_idx)+".0.npy"

# all_data = np.load("processed_trainval8020_March2020_256x256x24/"+str(institute), allow_pickle=True)

# 16 sites:
ids = ['TM', 'PH', 'TO', 'UT', 'DU',  'CP', 'IN', 'ST', 'SE', 'CG', 'NY', 'CH', 'GO', 'BO', 'KC', 'DY']
#val id = TK + AU
val_ids = ['TK', 'AU']
# select = [0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0]
#ids = ["UT","IN", "ST","SE","CG"]
train_datasets = []
n_trains = []
for id in ids:
    dir_ = "../brainseg_data_July17_FL/"
    all_data = np.load(dir_+f'{id}_data_uint8_train.npy', allow_pickle=True)

    all_data['train_x'] = all_data['xs_uint8'][:,:,:,:]
    all_data['train_y']=  all_data['label_classes'].copy()
    all_data['train_seg'] = all_data['ys_uint8'][:,:,:,:]

    #load the normals, randomly sample n samples where n is the same size as len(train_x), and concatenate
    #load the normals_tot (with keys train_x, train_y, train_seg),
    normals_tot = np.load(dir_+'normals_400x256x256x64_train_for_fl_64.npy', allow_pickle=True)
    # randomly sample n samples where n = len(all_data['train_x'])
    random_sample_normals = np.random.choice(normals_tot['train_x'].shape[0],
                                            all_data['train_x'].shape[0], replace=False)
    # concatenate the normals and the tumors
    all_data['train_x'] = np.concatenate((all_data['train_x'],
                                        normals_tot['train_x'][random_sample_normals]), axis=0)
    all_data['train_y'] = np.concatenate((all_data['train_y'],
                                        normals_tot['train_y'][random_sample_normals]), axis=0)
    all_data['train_seg'] = np.concatenate((all_data['train_seg'],
                                            normals_tot['train_seg'][random_sample_normals]), axis=0)
    n_train = len(all_data['train_x'])
    np.random.seed(42)
    idx_train = np.arange(n_train)

    xtrain = np.expand_dims(all_data['train_x'],4)
    ytrain2 = all_data['train_y']
    ytrain_roi =np.expand_dims(all_data['train_seg'],4)
    train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, all_data['train_y'], ytrain_roi  ))
    n_trains.append(n_train)
    train_dataset_ = train_dataset.map(rotate_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset_ = train_dataset_.shuffle(300, seed=global_starting_epoch).repeat().batch(BATCH_SIZE)
    train_datasets.append(train_dataset_)


    del all_data, normals_tot, xtrain, ytrain2, ytrain_roi
    

if (global_client_idx == 8):

    # all_data_val = np.load("processed_trainval8020_March2020_256x256x24/uint8_2023_train886_val227_256x256x64.npy", allow_pickle=True)
    if False:
        val_ids = ['TK', 'AU']
        all_data_val = {}
        first = True
        for val_id in val_ids:
            val_data = np.load(dir_+f'{val_id}_data_uint8.npy', allow_pickle=True)
            if first:
                all_data_val['val_x'] = val_data['xs_uint8'][:,:,:,8:8+48:2].astype(np.float32) / 255.0
                all_data_val['val_y']=  val_data['label_classes'].copy()
                all_data_val['val_seg'] = val_data['ys_uint8'][:,:,:,8:8+48:2]
                first = False
            else:
                all_data_val['val_x'] = np.concatenate((all_data_val['val_x'],
                                                    val_data['xs_uint8'][:,:,:,8:8+48:2].astype(np.float32) / 255.0), axis=0)
                all_data_val['val_y'] = np.concatenate((all_data_val['val_y'],
                                                        val_data['label_classes'].copy()), axis=0)
                all_data_val['val_seg'] = np.concatenate((all_data_val['val_seg'],
                                                        val_data['ys_uint8'][:,:,:,8:8+48:2]), axis=0)
    else:
       all_data_val = np.load(dir_+"combined_data_uint8_val.npy", allow_pickle=True)
       all_data_val['val_x'] = all_data_val['xs_uint8'][:,:,:,:]
       all_data_val['val_y']=  all_data_val['label_classes'].copy()
       all_data_val['val_seg'] = all_data_val['ys_uint8'][:,:,:,:]

       normals_val = np.load(dir_+'normals_867x256x256x64_val_for_fl_64.npy', allow_pickle=True)
       all_data_val['val_x'] = np.concatenate((all_data_val['val_x'], normals_val['val_x']), axis=0)
       all_data_val['val_y'] = np.concatenate((all_data_val['val_y'], normals_val['val_y']), axis=0)
       all_data_val['val_seg'] = np.concatenate((all_data_val['val_seg'], normals_val['val_seg']), axis=0)


    # all_data_val['val_x'] = all_data_val['val_x']
    xtest = np.expand_dims(all_data_val['val_x'],4)
    # ytest2 = all_data_val['val_y']
    # all_data_val['val_seg'] = all_data_val['val_seg']
    ytest_roi = np.expand_dims(all_data_val['val_seg'],4)

    n_test = len(all_data_val['val_x'])
    idx_test = np.arange(n_test)

    #ytest = test_data['ytrain']
    test_dataset = tf.data.Dataset.from_tensor_slices((xtest, all_data_val['val_y'], ytest_roi ) )
    del all_data_val, normals_val, xtest, ytest_roi





TRAIN_LENGTH = n_trains[0]
# BATCH_SIZE = 2
#BATCH_SIZE = 5

BUFFER_SIZE = 400
STEPS_PER_EPOCH = TRAIN_LENGTH // (BATCH_SIZE)
steps_per_epoch = []
num_steps = []
num_epochs = 1
for i in range(len(n_trains)):
    steps_per_epoch.append(n_trains[i]//BATCH_SIZE)
    num_steps.append(int(num_epochs * steps_per_epoch[i]))

# Counter({0: 1667, 3: 476, 4: 278, 2: 189, 1: 176})
# after testr emoval
# Counter({0: 1628, 3: 436, 4: 238, 2: 149, 1: 136})
# n = 2500
# 1628/2500 = 65%, 436/2500 = 18%, 238/2500 = 10%, 149/2500=6%, 136/2500=6%
# 65*0.10=6.5, 18*0.4=7.2, 10*0.7=6, 6*1.0=6, 6*1.0
def filter_out_balance(x,z,l):
  mb = tf.random.uniform(shape=[], maxval=11, dtype=tf.int32, seed=9)
  pilo = tf.random.uniform(shape=[], maxval=11, dtype=tf.int32, seed=4)
  normal = tf.random.uniform(shape=[], maxval=8, dtype=tf.int32, seed=24)
#   dipg = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=hvd.local_rank())
#   epend = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=hvd.local_rank())
#   return ((z == 1 and r > 8) or y == 0)
  return ( (z==0 and normal<2) or (z==3 and mb < 5) or (z==4 and pilo < 8) or z==2 or z==1 )

def apply_filter_fn(ds):
  return ds.filter(filter_out_balance)


# @tf.function
def test_map(ys,z, rois): #,label_of_plane_roi):
    ys = tf.cast(ys, tf.float32)
    ys = ys / 255.0
    z =  tf.cast(z, tf.float32)
    rois = tf.cast(rois, tf.float32)


    # ys = ys / 255.0

    # ys = ys[:,:,8:8+48:2,:]
#    label_of_plane_roi =label_of_plane_roi[16:]
    ys = tf.transpose(ys, perm=[2,0,1,3])
    print("test shape:", ys.shape)
    print(ys)
    rois = tf.transpose(rois, perm=[2,0,1,3])
    return ys, z, rois #, label_of_plane_roi

if (True or global_client_idx == 8):
    test_dataset_ = test_dataset.map(test_map)
    test_dataset_ = test_dataset_.batch(BATCH_SIZE_EVAL)

# train_dataset_ = train_dataset_.shuffle(BUFFER_SIZE, seed=global_starting_epoch).repeat().batch(BATCH_SIZE)

# train_dataset_ = train_dataset_.shuffle(BUFFER_SIZE, seed=hvd.local_rank()).apply(apply_filter_fn).batch(BATCH_SIZE).repeat()

# train_dataset_ = train_dataset_.shuffle(BUFFER_SIZE, seed=hvd.local_rank()).batch(BATCH_SIZE).repeat()
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# test_dataset_ = test_dataset_.batch(BATCH_SIZE)
# resampled_train_ds = train_dataset.batch(BATCH_SIZE).prefetch(2)
OUTPUT_CHANNELS = 1

def dice_coef(y_true, y_pred, smooth=1):
#     y_pred = tf.sigmoid(y_pred) > 0.5

    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    return tf.keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -tf.math.log(dice_coef(y_true, y_pred))


tf.keras.backend.clear_session()

model_0 = unet_model()
model_1 = unet_model()
model_2 = unet_model()
model_3 = unet_model()
model_4 = unet_model()
model_5 = unet_model()
model_6 = unet_model()
model_7 = unet_model()
model_8 = unet_model()
model_9 = unet_model()
model_10 = unet_model()
model_11 = unet_model()
model_12 = unet_model()
model_13 = unet_model()
model_14 = unet_model()
model_15 = unet_model()
def avg_weights(models, weights):
    # Check if the number of models matches the number of weights
    if len(models) != len(weights):
        raise ValueError("The number of models and weights don't match")
    
    # Extract the weights from each model
    model_weights = [model.get_weights() for model in models]

    # Check if all models have the same number of layers
    if not all(len(w) == len(model_weights[0]) for w in model_weights):
        raise ValueError("Not all models have the same number of layers")

    # Compute the weighted average of the weights
    average_weights = []
    for layer_weights in zip(*model_weights):
        average_weights.append(sum(w * weight for w, weight in zip(layer_weights, weights)))

    # Set the average weights to all models
    for model in models:
        model.set_weights(average_weights)

    return models


import scipy.io as io
tf.keras.backend.clear_session()
# print(model.summary())

start_epoch = 0
def loss(model, x, y, z, training):
    # y is class
    # z i s segmentation
    y_ = model(x, training=training)
    segmentation_loss = dice_coef_loss(z, y_[0])
    classification_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,
                                                            y_pred=y_[1],
                                                            from_logits=False)
    return strength_of_dice_loss*segmentation_loss + classification_loss, segmentation_loss, classification_loss


# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

#num_epochs = 1
epoch_loss_avg = tf.keras.metrics.Mean()
epoch_loss_avg2 = tf.keras.metrics.Mean()


if True and False:
    import tensorflow_addons as tfa
    f1_metric = tfa.metrics.F1Score(num_classes=5) # threshold is argmax
else:
    f1_metric = tf.keras.metrics.F1Score(
        average=None, threshold=None, name='f1_score', dtype=None
        )



#steps_per_epoch =  STEPS_PER_EPOCH // 1.0 + 1

#num_steps = int(num_epochs * steps_per_epoch)
#print(num_steps)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                            decay_steps=1000000,
                                                            decay_rate=0.5,
                                                            staircase=True)
optimizer_0 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_3 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_4 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_5 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_6 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_7 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_8 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_9 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_10 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_11 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_12 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_13 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_14 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_15 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
models = [model_0, model_1, model_2, model_3, model_4, model_5,model_6,model_7,model_8,model_9,model_10,model_11,model_12,model_13,model_14,model_15]
optimizers = [optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4, optimizer_5, \
         optimizer_6, optimizer_7, optimizer_8, optimizer_9, optimizer_10, optimizer_11, optimizer_12, optimizer_13, optimizer_14, optimizer_15, ]
#optimizer = tf.keras.optimizers.SGD(
#    learning_rate=0.0001,
#    momentum=0.2)
#optimizer = mixed_precision.LossScaleOptimizer(optimizer)

checkpoint = tf.train.Checkpoint(model=model_1, optimizer=optimizer_1)
train_losses_np = []
eval_losses_np = []
use_horovod = False



# initialize all weights to be the same:
if False and global_starting_epoch == 0:
    temp = [13.0,55,92,129,24,96,118,328/1.0,241,150,26,14,78,19,3,28]
    clients_scaling_factor = np.array(temp, dtype=float)
    tot_size = np.sum(clients_scaling_factor)
    clients_scaling_factor = clients_scaling_factor/tot_size
    models = avg_weights(models, clients_scaling_factor)

if True and global_starting_epoch == 0:
    file_ = "weights_from_Sites7and8/FL15_7and8First_H100_0_epoch50.h5"
    model_0.load_weights(file_)
    model_1.load_weights(file_)
    model_2.load_weights(file_)
    model_3.load_weights(file_)
    model_4.load_weights(file_)
    model_5.load_weights(file_)
    model_6.load_weights(file_)
    model_7.load_weights(file_)
    model_8.load_weights(file_)
    model_9.load_weights(file_)
    model_10.load_weights(file_)
    model_11.load_weights(file_)
    model_12.load_weights(file_)
    model_13.load_weights(file_)
    model_14.load_weights(file_)
    model_15.load_weights(file_)
fl_ckpt_path_1 = "checkpoints_3d_variation"+str(variation_idx)+"/epoch"+str(global_starting_epoch-1)+"global.h5"
#if global_starting_epoch > 0 :
#  model.load_weights(fl_ckpt_path_1)
  #model.load_model(fl_ckpt_path_1)
fl_ckpt_path_1 = "checkpoints_3d_variation"+str(variation_idx)+"/client_STSE" + "epochend"+str(global_starting_epoch-1)+".h5"
if global_starting_epoch > 0 :
  model_0.load_weights(fl_ckpt_path_1)
  model_1.load_weights(fl_ckpt_path_1)
  model_2.load_weights(fl_ckpt_path_1)
  model_3.load_weights(fl_ckpt_path_1)
  model_4.load_weights(fl_ckpt_path_1)
  model_5.load_weights(fl_ckpt_path_1)
  model_6.load_weights(fl_ckpt_path_1)
  model_7.load_weights(fl_ckpt_path_1)
  model_8.load_weights(fl_ckpt_path_1)
  model_9.load_weights(fl_ckpt_path_1)
  model_10.load_weights(fl_ckpt_path_1)
  model_11.load_weights(fl_ckpt_path_1)
  model_12.load_weights(fl_ckpt_path_1)
  model_13.load_weights(fl_ckpt_path_1)
  model_14.load_weights(fl_ckpt_path_1)
  model_15.load_weights(fl_ckpt_path_1)


if  mu > 0.0:

    # fed prox:
    global_model = unet_model()
    global_model.set_weights(model_0.get_weights())
    #global_model.load_weights(fl_ckpt_path_1)
    # w_star_list = [layer.get_weights()[0] for layer in global_model.layers if layer.get_weights()]
    # w_star_list = [layer.trainable_weights[0] for layer in global_model.layers if layer.trainable_weights]
    w_star_list = [tf.constant(layer.trainable_weights[0].numpy()) for layer in global_model.layers if layer.trainable_weights]

    #w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
    # mu = 0.1
    @tf.function
    def custom_loss(w_star_list, w_local_list):
       total_loss = 0.0
       for i in range(len(w_local_list)):
           diff = w_local_list[i] - (w_star_list[i])
           loss_add = tf.reduce_sum(tf.square(diff))
           total_loss += loss_add

       # Compute the combined loss: cross_entropy + mu * ||w - w_star||^2
       return  mu * total_loss



@tf.function
def training_step2(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,
                         images,
                         labels,
                         roi,
                         training=True)
#       loss_value = optimizer.get_scaled_loss(loss_value)
       if mu > 0.0:
         w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
         loss_value += custom_loss(w_star_list, w_local_list)
   if (use_horovod):
       tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, model.trainable_variables)
#   grads = optimizer.get_unscaled_gradients(grads)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

   if (first_batch == 0 and use_horovod):
     hvd.broadcast_variables(model.variables, root_rank=0)
     hvd.broadcast_variables(optimizer.variables(), root_rank=0)
   return loss_value,  seg_loss, class_loss

@tf.function
def training_step2_1(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,
                         images,
                         labels,
                         roi,
                         training=True)
#       loss_value = optimizer.get_scaled_loss(loss_value)
       if mu > 0.0:
         w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
         loss_value += custom_loss(w_star_list,w_local_list)
   if (use_horovod):
       tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, model.trainable_variables)
#   grads = optimizer.get_unscaled_gradients(grads)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

   if (first_batch == 0 and use_horovod):
     hvd.broadcast_variables(model.variables, root_rank=0)
     hvd.broadcast_variables(optimizer.variables(), root_rank=0)
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_2(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,
                         images,
                         labels,
                         roi,
                         training=True)
#       loss_value = optimizer.get_scaled_loss(loss_value)
       if mu > 0.0:
         w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
         loss_value += custom_loss(w_star_list,w_local_list)
   if (use_horovod):
       tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, model.trainable_variables)
#   grads = optimizer.get_unscaled_gradients(grads)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

   if (first_batch == 0 and use_horovod):
     hvd.broadcast_variables(model.variables, root_rank=0)
     hvd.broadcast_variables(optimizer.variables(), root_rank=0)
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_3(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,
                         images,
                         labels,
                         roi,
                         training=True)
#       loss_value = optimizer.get_scaled_loss(loss_value)
       if mu > 0.0:
         w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
         loss_value += custom_loss(w_star_list,w_local_list)
   if (use_horovod):
       tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, model.trainable_variables)
#   grads = optimizer.get_unscaled_gradients(grads)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

   if (first_batch == 0 and use_horovod):
     hvd.broadcast_variables(model.variables, root_rank=0)
     hvd.broadcast_variables(optimizer.variables(), root_rank=0)
   return loss_value,  seg_loss, class_loss

@tf.function
def training_step2_4(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_5(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_6(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_7(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_8(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss

@tf.function
def training_step2_9(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_10(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_11(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_12(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss

@tf.function
def training_step2_13(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_14(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0: 
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss
@tf.function
def training_step2_15(model, optimizer, images, labels,  roi, first_batch, w_star_list=None):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,images,labels,roi,training=True)
       if mu > 0.0:
           w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
           loss_value += custom_loss(w_star_list,w_local_list)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   return loss_value,  seg_loss, class_loss


def evaluate_step(model, val_dataset):
   num_eval_steps = n_test // BATCH_SIZE_EVAL
   losses = 0.0
   classification_loss = 0.0
   for (batch, (x,  z, roi)) in enumerate(val_dataset.take(num_eval_steps)):
     print("batch", batch, "x.shape:", x.shape, "z.shape", z.shape)
     logits = model(x, training=False)
#      test_accuracy.update_state(z, logits)
     loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=z,
                                                           y_pred=logits[1],
                                                           from_logits=False)

     depth = 5
     zz  =z.numpy()
     logits_np = logits[1].numpy()
     # print(zz, logits_np)
     z2 = tf.one_hot(zz, depth)
     f1_metric.update_state(z2, logits_np)
   return loss,  f1_metric.result().numpy()  #, test_accuracy.result()



epochs =  []
dice_scores_epoch = []
f1_nps = []


#models = [model_0, model_1, model_2, model_3, model_4, model_5,model_6,model_7,model_8,model_9,model_10,model_11,model_12,model_13,model_14,model_15]
local_epochs = 0
if mu == 0.0:
    w_star_list = None
for local_epoch in range(100):
    # 1 local epoch
    epoch = local_epoch + global_starting_epoch
    for i in range(len(train_datasets)):
        for batch, (x, z, roi) in enumerate(train_datasets[i].take(num_steps[i]//1)):
            #epoch = int(batch//steps_per_epoch[i]) +global_starting_epoch

            if i ==0:            
                class_loss, seg_loss_comp, class_loss_comp = training_step2(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==1:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_1(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==2:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_2(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==3:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_3(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==4:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_4(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==5:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_5(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==6:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_6(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==7:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_7(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==8:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_8(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==9:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_9(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==10:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_10(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==11:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_11(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==12:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_12(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==13:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_13(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==14:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_14(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)
            elif i==15:
                class_loss, seg_loss_comp, class_loss_comp = training_step2_15(models[i], optimizers[i], x, z, roi, batch == 0, w_star_list)

        if i == 8:
            eval_loss_np, f1_score_np = evaluate_step(models[i], test_dataset_)
            f1_metric.reset_states()

            eval_losses_np.append(eval_loss_np)
            eval_loss_np_2 = eval_loss_np.numpy()
            dice_scores_epoch.append(eval_loss_np_2)
            row = "epoch: " + str(epoch) + " train loss:" + str(epoch_loss_avg.result().numpy()  ) + " class train loss: " + str(epoch_loss_avg2.result().numpy() ) \
                    + " eval class loss: "+ str(f1_score_np) #+ " eval acc: " + str(acc)
            print(row)
        #    row_ = checkpoint_path + " " +row
            with open("testsetresults_3d" + str(global_client_idx) +"synth" + str(use_only_synths) +  ".csv",'a') as fd:
                fd.write(row)
                fd.write("\n")

    fl_ckpt_path_2 = "checkpoints_3d_variation"+str(variation_idx)+"/client_STSE" + "epochend"+str(epoch)+".h5"
    model_0.save(fl_ckpt_path_2,  include_optimizer=False)


    # scaling factors for the weighted average of the parameters.
    temp = [13.0,55,92,129,24,96,118,328/1.0,241,150,26,14,78,19,3,28]
    #temp = num_steps
    clients_scaling_factor = np.array(temp, dtype=float)
    #clients_scaling_factor = np.array(temp)
    tot_size = np.sum(clients_scaling_factor)
    clients_scaling_factor = clients_scaling_factor/tot_size

    models = avg_weights(models, clients_scaling_factor)
    # Proximal term:
    if mu > 0.0:
        global_model.set_weights(models[0].get_weights())
        w_star_list = [tf.constant(layer.trainable_weights[0].numpy()) for layer in global_model.layers if layer.trainable_weights]

str_ = "_iiddata_"+str(use_iid_data)+"_nummicrobatches_"+str(num_microbatches)+"_batchsize_"+str(BATCH_SIZE)+"_fp16_"+str(use_fp16)
# fl_ckpt_path_2 = "checkpoints_3d_cds"+"/dpsgd_cds_epoch_"+str(epoch)+"_l2normclip_"+str(l2_norm_clip)+"_noisemult_"+str(noise_multiplier)+str_+".h5"
fl_ckpt_path_2 = "checkpoints_3d_variation"+str(variation_idx)+"/client_STSE" + "epochend"+str(global_starting_epoch)+".h5"
#fl_ckpt_path_2 = "checkpoints_3d/client"+str(global_client_idx)+"epochend"+str(global_epoch+global_starting_epoch)+".h5"

#model_0.save(fl_ckpt_path_2,  include_optimizer=False)



