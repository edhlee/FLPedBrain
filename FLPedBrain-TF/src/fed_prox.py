import numpy as np
# redundant / change
import tensorflow as tf
import tensorflow 
import sys
import pickle
print(tf.__version__)
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input,  Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow import keras as K
# from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from i3d_inception import Inception_Inflated3d
from unet_model_ import unet_model
import scipy.io as io



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

tf.compat.v1.enable_eager_execution()

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

use_fp16 = sys.argv[3].lower() == 'true'

global_test_index = int(sys.argv[1])
np.random.seed(42)
global_offset_gold = int(sys.argv[4])
threshold_gold = float(sys.argv[5])
use_all_data = sys.argv[6].lower() == 'true'
global_starting_epoch = int(sys.argv[7])
global_client_idx = int(sys.argv[8])
global_num_clients = int(sys.argv[9])
global_num_local_epochs = int(sys.argv[10])
use_iid_data = sys.argv[11].lower() == 'true'
K = global_num_clients
client_frac = float(sys.argv[12])
num_active_clients = int(max(K*client_frac, 1.0))
chance_of_active_client = num_active_clients / K
use_client = np.random.binomial(1, chance_of_active_client, 1) > 0.5
variation_idx = int(sys.argv[13])
site_split_idx = int(sys.argv[14])
l2_norm_clip = float(sys.argv[15])
noise_multiplier = float(sys.argv[16])
num_microbatches = int(sys.argv[17])
use_only_synths = sys.argv[18].lower() == 'true'
strength_of_dice_loss = 0.1
BATCH_SIZE = 2
BATCH_SIZE_EVAL = 8

institute = "institute_" + str(global_client_idx) + ".0.npy"

ids = ['TM', 'PH', 'TO', 'UT', 'DU',  'CP', 'IN', 'ST', 'SE', 'CG', 'NY', 'CH', 'GO', 'BO', 'KC', 'DY']
val_ids = ['TK', 'AU']

id = ids[global_client_idx]
dir_ = "../brainseg_data_July17_FL/"
all_data = np.load(dir_+f'{id}_data_uint8_train.npy', allow_pickle=True)





if (use_only_synths):
    synth_data = np.load("synth/0ckptsite_checkpoint_synth_from_true_class_0.npy", allow_pickle=True)
    tot = np.concatenate((synth_data['1'], synth_data['2'], synth_data['3'], synth_data['4']), axis=0)
    tot = tot.astype(np.float32) / 255.0
    tot_labels = np.concatenate((np.ones(synth_data['1'].shape[0]), np.ones(synth_data['2'].shape[0])*2,
                                 np.ones(synth_data['3'].shape[0])*3, np.ones(synth_data['4'].shape[0])*4), axis=0)

    # print(tot.shape)
    # print(tot.dtype)

    # print(tot_labels.shape)
    # print(tot_labels.dtype)


# subsample to 24 slices from 64

all_data['train_x'] = all_data['xs_uint8'][:,:,:,:]
# all_data['val_x'] = all_data['val_x'][:,:,:,8:8+48:2]
all_data['train_y']=  all_data['label_classes'].copy()
all_data['train_seg'] = all_data['ys_uint8'][:,:,:,:]
# all_data['val_seg'] = all_data['val_seg'][:,:,:,8:8+48:2]

#load the normals, randomly sample n samples where n is the same size as len(train_x), and concatenate
#load the normals_tot (with keys train_x, train_y, train_seg),
normals_tot = np.load(dir_+'normals_400x256x256x64_train_for_fl_64.npy', allow_pickle=True)
# randomly sample n samples where n = len(all_data['train_x'])
random_sample_normals = np.random.choice(normals_tot['train_x'].shape[0],
                                         all_data['train_x'].shape[0]//2, replace=False)

# concatenate the normals and the tumors
all_data['train_x'] = np.concatenate((all_data['train_x'],
                                      normals_tot['train_x'][random_sample_normals]), axis=0)
all_data['train_y'] = np.concatenate((all_data['train_y'],
                                      normals_tot['train_y'][random_sample_normals]), axis=0)
all_data['train_seg'] = np.concatenate((all_data['train_seg'],
                                        normals_tot['train_seg'][random_sample_normals]), axis=0)


if (use_only_synths):
    all_data['train_x'] = np.concatenate((all_data['train_x'], tot), axis=0)
    all_data['train_y'] = np.concatenate((all_data['train_y'], tot_labels), axis=0)
    all_data['train_seg'] = np.concatenate((all_data['train_seg'], tot*0), axis=0)


# train_data = pickle.load(pickle_in)
n_train = len(all_data['train_x'])
np.random.seed(42)
idx_train = np.arange(n_train)


xtrain = []
ytrain = []
xtest = []
ytest = []
ytrain2= []
ytest2 = []

xtrain = np.expand_dims(all_data['train_x'],4)
ytrain2 = all_data['train_y']
# ytrain = train_data['ytrain']
ytrain_roi =np.expand_dims(all_data['train_seg'],4)
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, all_data['train_y'], ytrain_roi  ))

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


@tf.function
def rotate_tf(image, c, roi):

    image = tf.cast(image, tf.float32)

    image /= 255.0
    c =  tf.cast(c, tf.float32)
    roi = tf.cast(roi,tf.float32)

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
        image_2d = tf.image.resize(image_2d, [256,256], preserve_aspect_ratio=False,antialias=False, name=None)
        image_mask_2d = tf.image.resize(image_mask_2d, [256,256], preserve_aspect_ratio=False,antialias=False, name=None)
        ys.append(image_2d)
        ys_mask.append(image_mask_2d)

    ys = tf.convert_to_tensor(ys, dtype=tf.float32)
#     label_of_plane_roi = tf.convert_to_tensor(label_of_plane_roi,  dtype=tf.float32)
    ys_mask = tf.convert_to_tensor(ys_mask, dtype=tf.float32)
    return ys,  c, ys_mask




TRAIN_LENGTH = n_train
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // (BATCH_SIZE)

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
    ys = tf.transpose(ys, perm=[2,0,1,3])
    rois = tf.transpose(rois, perm=[2,0,1,3])

    return ys, z, rois #, label_of_plane_roi

# train_dataset_=train_dataset.filter(filter_out_balance)
train_dataset_ = train_dataset.map(rotate_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if (global_client_idx == 8):
    test_dataset_ = test_dataset.map(test_map)
    test_dataset_ = test_dataset_.batch(BATCH_SIZE_EVAL)

train_dataset_ = train_dataset_.shuffle(BUFFER_SIZE, seed=global_starting_epoch).repeat().batch(BATCH_SIZE)

OUTPUT_CHANNELS = 1

def dice_coef(y_true, y_pred, smooth=1):
#     y_pred = tf.sigmoid(y_pred) > 0.5

    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    return tf.keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def binary_focal_loss_fixed_2(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma=2.
    alpha=.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

#         epsilon = K.epsilon()
    epsilon =  1e-12
    # clip to prevent NaN's and Inf's
    pt_1 = tf.keras.backend.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.keras.backend.clip(pt_0, epsilon, 1. - epsilon)
    temp = -1000.0*tf.keras.backend.mean(alpha * tf.keras.backend.pow(1. - pt_1, gamma) *  tf.keras.backend.log(pt_1))           -1000.0*tf.keras.backend.mean((1 - alpha) * tf.keras.backend.pow(pt_0, gamma) *  tf.keras.backend.log(1. - pt_0))
    return temp
def dice_coef(y_true, y_pred, smooth=1):
#     y_pred = tf.sigmoid(y_pred) > 0.5

    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    return tf.keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -tf.math.log(dice_coef(y_true, y_pred))

def dice_coef_loss2(y_true, y_pred):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    axis=(1, 2, 3)
    smooth=0.1
    intersection = tf.keras.backend.sum(y_pred * y_true, axis=axis)
    p = tf.keras.backend.sum(y_pred, axis=axis)
    t = tf.keras.backend.sum(y_true, axis=axis)
    numerator = tf.keras.backend.sum(intersection + smooth)
    denominator = tf.keras.backend.mean(t + p + smooth)
    dice_loss = - tf.keras.backend.log(2.*numerator) + tf.keras.backend.log(denominator)

    return dice_loss / 20.0


model = unet_model()

tf.keras.backend.clear_session()
print(model.summary())

# model.load_weights("checkpoints_new_dice/experiment_epoch_33.ckpt")
# model.load_weights("checkpoints_new_dice/experiment_epoch_33.ckpt")
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

num_epochs = 5
epoch_loss_avg = tf.keras.metrics.Mean()
epoch_loss_avg2 = tf.keras.metrics.Mean()


if True:
    import tensorflow_addons as tfa
    f1_metric = tfa.metrics.F1Score(num_classes=5) # threshold is argmax
else:
    f1_metric = tf.keras.metrics.F1Score(
        average=None, threshold=None, name='f1_score', dtype=None
        )



steps_per_epoch =  STEPS_PER_EPOCH // 1.0 + 1

num_steps = int(num_epochs * steps_per_epoch)
print(num_steps)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                            decay_steps=100*steps_per_epoch,
                                                            decay_rate=0.5,
                                                            staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.SGD(
#    learning_rate=0.0001,
#    momentum=0.2)
#optimizer = mixed_precision.LossScaleOptimizer(optimizer)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
train_losses_np = []
eval_losses_np = []
use_horovod = False



# initialize all weights to be the same:
if global_starting_epoch == 0:
  model.load_weights("initial_weights_unetmodel.h5")


fl_ckpt_path_1 = "checkpoints_3d_variation"+str(variation_idx)+"/epoch"+str(global_starting_epoch-1)+"global.h5"
#if global_starting_epoch > 0 :
#  model.load_weights(fl_ckpt_path_1)
  #model.load_model(fl_ckpt_path_1)


mu=0.1
if global_starting_epoch > 0 and mu > 0.0:

    # fed prox:
    global_model = unet_model()
    global_model.load_weights(fl_ckpt_path_1)
    # w_star_list = [layer.get_weights()[0] for layer in global_model.layers if layer.get_weights()]
    # w_star_list = [layer.trainable_weights[0] for layer in global_model.layers if layer.trainable_weights]
    w_star_list = [tf.constant(layer.trainable_weights[0].numpy()) for layer in global_model.layers if layer.trainable_weights]

    w_local_list = [layer.trainable_weights[0] for layer in model.layers if layer.trainable_weights]
    # mu = 0.1
    @tf.function
    def custom_loss():
       total_loss = 0.0
       for i in range(len(w_local_list)):
           diff = w_local_list[i] - (w_star_list[i])
           loss_add = tf.reduce_sum(tf.square(diff))
           total_loss += loss_add

       # Compute the combined loss: cross_entropy + mu * ||w - w_star||^2
       return  mu * total_loss



@tf.function
def training_step2(images, labels,  roi, first_batch):
   with tf.GradientTape() as tape:
       loss_value, seg_loss, class_loss = loss(model,
                         images,
                         labels,
                         roi,
                         training=True)
#       loss_value = optimizer.get_scaled_loss(loss_value)
       if mu > 0.0:
         loss_value += custom_loss()
   if (use_horovod):
       tape = hvd.DistributedGradientTape(tape)
   grads = tape.gradient(loss_value, model.trainable_variables)
#   grads = optimizer.get_unscaled_gradients(grads)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

   if (first_batch == 0 and use_horovod):
     hvd.broadcast_variables(model.variables, root_rank=0)
     hvd.broadcast_variables(optimizer.variables(), root_rank=0)
   return loss_value,  seg_loss, class_loss


def evaluate_step(val_dataset):
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
for (batch, (x,  z, roi)) in enumerate(train_dataset_.take(num_steps) ):
    # Optimize the model
    epoch = int(batch//steps_per_epoch)
    epochs.append(epoch)
    if (batch == 0 and epoch == 0 and use_horovod):
        first_batch=True
    else: first_batch = False

    # test_accuracy = tf.keras.mtrics.SparseCategoricalAccuracy(name='test_accuracy')
    #loss_value = training_step(x, y,z, batch == 0)
    class_loss, seg_loss_comp, class_loss_comp = training_step2(x, z, roi, batch == 0)
    print("batch", batch, "global_client", str(global_client_idx),
            "loss_tot", class_loss,
            "seg_loss_comp", seg_loss_comp,
            "class_loss_comp", class_loss_comp)

    loss_value = class_loss
    epoch_loss_avg(loss_value)  # Add current batch loss
    epoch_loss_avg2(class_loss)
    train_losses_np.append(epoch_loss_avg.result().numpy())
    #    if ( (batch % 10) ==0  and  hvd.local_rank() == 0):
    if ( (batch % 10) ==0 ):

        print("epoch: ", str(global_starting_epoch), " loss:", str(epoch_loss_avg.result() ) )
    if ( (batch % steps_per_epoch == 0) ):
        row = "epoch: " + str(epoch) + " train loss:" + str(epoch_loss_avg.result().numpy()  ) + " class train loss: " + str(epoch_loss_avg2.result().numpy() )

        if (global_starting_epoch % 1 == 0 and  global_client_idx == 8):
            eval_loss_np, f1_score_np = evaluate_step(test_dataset_)
            f1_metric.reset_states()

            eval_losses_np.append(eval_loss_np)
            eval_loss_np_2 = eval_loss_np.numpy()
            dice_scores_epoch.append(eval_loss_np_2)

            #classification_loss_np_2 = classification_loss_np.numpy()

            row = "epoch: " + str(epoch) + " train loss:" + str(epoch_loss_avg.result().numpy()  ) + " class train loss: " + str(epoch_loss_avg2.result().numpy() ) \
                    + " eval class loss: "+ str(f1_score_np) #+ " eval acc: " + str(acc)

            print(row)

                #str_ = "_iiddata_"+str(use_iid_data)+"_nummicrobatches_"+str(num_microbatches)+"_batchsize_"+str(BATCH_SIZE)+"_fp16_"+str(use_fp16)
        # fl_ckpt_path_2 = "checkpoints_3d_variation"+str(variation_idx)+"/synth"+str(use_only_synths)+"epochend"+str(epoch)+".h5"
        # model.save(fl_ckpt_path_2,  include_optimizer=False)

    #    row_ = checkpoint_path + " " +row
        with open("testsetresults_3d" + str(global_client_idx) +"synth" + str(use_only_synths) +  ".csv",'a') as fd:
            fd.write(row)
            fd.write("\n")

        if global_starting_epoch % 1 == 0 and global_client_idx == 8:
            all_data = {}
            all_data['epochs'] = np.array(epochs)
                # all_data['f1_weighted'] = np.array(f1_scores_weighted)
                # all_data['f1'] = np.array(f1_scores)
            # f1_score_np=0.0
            f1_nps.append(f1_score_np)
            all_data['f1'] = np.array(f1_nps)

            with open( "evals_npy/"+str(global_client_idx) +"synth" + str(use_only_synths)  + '_sites.npy', 'wb') as filehandle:
                temp = pickle.dump(all_data, filehandle, protocol=4)

    #         model.save_weights(checkpoint_path)
    #         train_loss_results.append(epoch_loss_avg.result())


str_ = "_iiddata_"+str(use_iid_data)+"_nummicrobatches_"+str(num_microbatches)+"_batchsize_"+str(BATCH_SIZE)+"_fp16_"+str(use_fp16)
# fl_ckpt_path_2 = "checkpoints_3d_cds"+"/dpsgd_cds_epoch_"+str(epoch)+"_l2normclip_"+str(l2_norm_clip)+"_noisemult_"+str(noise_multiplier)+str_+".h5"
fl_ckpt_path_2 = "checkpoints_3d_variation"+str(variation_idx)+"/client"+str(global_client_idx)+ "sitesplitidx" + str(site_split_idx) + "epochend"+str(global_starting_epoch)+".h5"
#fl_ckpt_path_2 = "checkpoints_3d/client"+str(global_client_idx)+"epochend"+str(global_epoch+global_starting_epoch)+".h5"

model.save(fl_ckpt_path_2,  include_optimizer=False)





