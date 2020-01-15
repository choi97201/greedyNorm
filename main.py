# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from numpy import expand_dims
import os
import shutil
import argparse
import numpy as np

from keras.utils import multi_gpu_model
from keras.models import Model

from model import UnetModel, FCNModel, Vgg16Model, DeepLabv3plus, Adversarial_1, Adversarial_2, Unet_Attention, Unet_Attention_BN, Unet_Attention_BIN
from callbacks import *
from loss import *
from metrics import *
from load_data import *
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL = "FCN"
DATASET = "voc"
METHOD = "normal"
EPOCHS = 100
INPUT_SHAPE = 512
LEARNING_RATE = 0.007
BATCH_SIZE = 16
SEED = 777

np.random.seed(SEED)

# def visualize_conv_layer(model,layer_name):
#     layer_output = model.get_layer(layer_name).output
#     print("success")
#     intermediate_model = Model(inputs=model.input, outputs=layer_output)
#     print("success")
#     intermediate_prediction = intermediate_model.predict(x_train[2].reshape(1, 28, 28, 1))
#     print("success")
#     row_size = 4
#     col_size = 8
#     img_index = 0
#     print(np.shape(intermediate_prediction))
#     fig, ax = plt.subplots(row_size, col_size, figsize=(10, 8))
#     for row in range(0, row_size):
#         for col in range(0, col_size):
#             ax[row][col].imshow(intermediate_prediction[0, :, :, img_index], cmap='gray')
#             img_index = img_index + 1

'''shloss.py
if args.method == "normal":
    if args.loss_fn == "orgin":
        loss = categorical_croosentropy(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_convex":
        loss = categorical_crossentropy_regularization_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_linear":
        loss = categorical_croosentropy_regularization_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_exp_convex":
        loss = categorical_crossentropy_expotential_regularization_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_exp_linear":
        loss = categorical_crossentropy_expotential_regularization_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_exp_comb_convex":
        loss = categorical_crossentropy_expotential_regularization_combination_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_exp_comb_linear":
        loss = categorical_crossentropy_expotential_regularization_combination_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_exp_comb_aver_convex":
        loss = categorical_crossentropy_expotential_average_regularization_combination_exp_combination_conex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "orgin_exp_comb_aver_linear":
        loss = categorical_crossentropy_expotential_average_regularization_combination_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge":
        loss = categorical_hinge(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge_convex":
        loss = categorical_hinge_regularization_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge_linear":
        loss = categorical_hinge_regularization_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge_exp_convex":
        loss = categorical_hinge_expotential_regularization_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge_exp_linear":
        loss = categorical_hinge_expotential_regularization_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge_exp_comb_convex":
        loss = categorical_hinge_expotential_regularization_combination_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "hinge_exp_comb_aver_convex":
        loss = categorical_hinge_expotential_average_regularization_combination_exp_combination_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal":
        loss = categorical_focal(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal_convex":
        loss = categorical_focal_regularization_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal_linear":
        loss = categorical_focal_regularization_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal_exp_convex":
        loss = categorical_focal_expotential_regularization_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal_exp_linear":
        loss = categorical_focal_expotential_regularization_linear(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal_exp_comb_convex":
        loss = categorical_focal_expotential_regularization_combination_convex(l1=args.l1, l2=args.l2)
    elif args.loss_fn == "focal_exp_comb_linear":
        loss = categorical_focal_expotential_regularization_combination_linear(l1=args.l1, l2=args.l2)
'''


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument("--model", type=str, default=MODEL,
                        metavar="FCN/ FCNBN / FCNIN / FCNGN / FCNLN /FCNSN / FCNBIN / FCNBGN / FCNBSN / FCNBLN / Unet / UnetBN / UnetIN / UnetGN / UnetLN / UnetSN / UnetBIN / UnetBGN / UnetBLN / UnetBSN /AttentionUnet / AttentionUnetBN / AttentionUnetBIN / v3 / v3BN / v3BIN",
                        help="Choose your model")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        metavar="voc / atr / cifar10",
                        help="Choose dataset which will be used")
    parser.add_argument("--method", type=str, default=METHOD,
                        metavar="normal / adversarial",
                        help="Choose training method you will use")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Enter epochs you want")
    parser.add_argument("--input-shape", type=int, default=INPUT_SHAPE,
                        help="Enter input shape you want")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Enter learning rate you want")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Enter batch size you want")
    parser.add_argument("--l1", type=float, default=0.,
                        metavar="",
                        help="Set L1 Loss weight")
    parser.add_argument("--l2", type=float, default=0.,
                        metavar="",
                        help="Set L2 Loss weight")
    parser.add_argument("--multigpu", type=int, default=0,
                        help="Set number of gpus you will use")
    parser.add_argument("--loss-fn", type=str, default="orgin",
                        metavar="orgin / focal / hinge / orgin_focal / hinge_focal / focal_focal / orgin_exp / focal_exp / hinge_exp / orgin_focal_exp / hinge_focal_exp / focal_focal_exp",
                        help="Choose your loss function")
    parser.add_argument("--version", type=int, default=1,
                        help="Enter version")
    parser.add_argument("--checkpoint", type=str, default=None,
                        metavar="/path/to/weights/hdf5",
                        help="Path to weight .hdf5 file")
    parser.add_argument("--init-epoch", type=int, default=0,
                        help="If you use trained weight")
    parser.add_argument("--multiprocessing", type=int, default=0,
                        help="If you want to use multiprocessing when training")
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers you want")
    parser.add_argument("--divide-param", type=int, default=1,
                        help="Set number for dividing parameter")
    return parser.parse_args()

def main():
    args = get_arguments()
    assert args.command in ['train', 'test']

    print("━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━")
    print("  Command            ┃   " + args.command)
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Model              ┃   " + args.model)
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Dataset            ┃   " + args.dataset)
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Method             ┃   " + args.method)
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Epochs             ┃   " + str(args.epochs))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Learning Rate      ┃   " + str(args.learning_rate))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Batch Size         ┃   " + str(args.batch_size))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  L1                 ┃   " + str(args.l1))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  L2                 ┃   " + str(args.l2))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Multi GPU          ┃   " + str(args.multigpu))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Loss Function      ┃   " + args.loss_fn)
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Version            ┃   " + str(args.version))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Initial Epoch      ┃   " + str(args.init_epoch))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  MultiProcessing    ┃   " + str(args.multiprocessing))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Workers            ┃   " + str(args.workers))
    print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━")
    print("  Dividing Number    ┃   " + str(args.divide_param))
    print("━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━")

    if args.dataset == "voc":
        classes = 21
        trainset = [line.rstrip() for line in open('./voc_trainaug.txt', 'r')]
        valset = [line.rstrip() for line in open('./voc_val.txt', 'r')]

    elif args.dataset == "mnist":
        classes = 10

    elif args.dataset == "atr":
        classes = 18
        trainset = [line.rstrip() for line in open('./atr_train.txt', 'r')]
        valset = [line.rstrip() for line in open('./atr_val.txt', 'r')]

    print("TRAIN DATA: {}".format(len(trainset)))
    print("TEST DATA: {}".format(len(valset)))

    if args.command == "train":
        '''
        Set Callbacks
        '''
        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')

        checkpoint_dir = os.path.join('./checkpoint', args.model)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        if not os.path.isdir('./history'):
            os.mkdir('./history')

        if args.dataset == 'atr':
            if not os.path.isdir('./history/atr'):
                os.mkdir('./history/atr')
            history_dir = os.path.join('./history/atr', args.model)
            if not os.path.isdir(history_dir):
                os.mkdir(history_dir)

        elif args.dataset == 'voc':
            if not os.path.isdir('./history/voc'):
                os.mkdir('./history/voc')
            history_dir = os.path.join('./history/voc', args.model)
            if not os.path.isdir(history_dir):
                os.mkdir(history_dir)

        if not os.path.isdir('./logs'):
            os.mkdir('./logs')

        logs_dir = os.path.join('./logs', args.model)
        if not os.path.isdir(logs_dir):
            os.mkdir(logs_dir)

        callback_name = '{}_{}_{}_{}_{}_{}'.format(args.loss_fn, args.batch_size, args.learning_rate, args.l1, args.l2,
                                                   args.version)
        cp = callback_checkpoint(filepath=os.path.join(checkpoint_dir, callback_name + '_{epoch:04d}.hdf5'),
                                 monitor='val_iou',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True,
                                 save_weights_only=True)

        lr = callback_lrscheduler(initial_lrate=args.learning_rate,
                                  max_epoch=args.epochs)

        tb = callback_tensorboard(log_dir=logs_dir + '/' + callback_name,
                                  batch_size=args.batch_size)

        batch_cl = callback_customcsvlogger(filename=history_dir + '/' + callback_name + '_batch_log.csv',
                                            separator=',', append=True)
        epoch_cl = callback_csvlogger(filename=history_dir + '/' + callback_name + '_' + str(args.loss_fn) + '_' + str(
                args.batch_size) + '_epoch_log.csv', separator=',', append=True)

        es = callback_earlystopping(monitor='val_iou', patience=5)

        callback_list = [cp, lr, tb, batch_cl, epoch_cl, es]

        if args.multiprocessing:
            print("####### Use multiprocessing on generator #######")
            use_multiprocessing = True
            workers = args.workers
        else:
            use_multiprocessing = False
            workers = 1

        train_generator = DataGenerator(args.dataset, trainset, args.batch_size, args.input_shape, 'train', classes)
        val_generator = DataGenerator(args.dataset, valset, args.batch_size, args.input_shape, 'val', classes)

        ''' loss.py'''
        if args.method == "normal":
            if args.loss_fn == "orgin":
                loss = categorical_crossentropy(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "focal":
                loss = categorical_focal(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "hinge":
                loss = categorical_hinge(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "orgin_focal":
                loss = categorical_crossentropy_focal(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "hinge_focal":
                loss = categorical_hinge_focal(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "focal_focal":
                loss = categorical_focal_focal(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "orgin_exp":
                loss = categorical_crossentropy_expotential_regularization(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "hinge_exp":
                loss = categorical_hinge_expotential_regularization(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "focal_exp":
                loss = categorical_focal_expotential_regularization(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "orgin_focal_exp":
                loss = categorical_crossentropy_focal_expotential_regularization(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "hinge_focal_exp":
                loss = categorical_hinge_focal_expotential_regularization(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "focal_focal_exp":
                loss = categorical_focal_focal_expotential_regularization(l1=args.l1, l2=args.l2)
            if args.model == "v3":
                model = DeepLabv3plus(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "v3BN":
                model = DeepLabv3plus_BN(input_shape=(arg.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "v3BIN":
                model = DeepLabv3plus_BIN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "FCN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "FCNBN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm2=True)
            elif args.model == "FCNIN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_instancenorm=True)
            elif args.model == "FCNGN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_groupnorm=True)
            elif args.model == "FCNLN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_layernorm=True)
            elif args.model == "FCNSN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_spectralnorm=True)
            elif args.model == "FCNBBN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_batchnorm2=True)
            elif args.model == "FCNBIN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_instancenorm=True)
            elif args.model == "FCNBGN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_groupnorm=True)
            elif args.model == "FCNBLN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_layernorm=True)
            elif args.model == "FCNBSN":
                model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_spectralnorm=True)
            elif args.model == "Unet":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "UnetBN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm2=True)
            elif args.model == "UnetIN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_instancenorm=True)
            elif args.model == "UnetGN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_groupnorm=True)
            elif args.model == "UnetLN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_layernorm=True)
            elif args.model == "UnetSN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_spectralnorm=True)
            elif args.model == "UnetBBN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_batchnorm2=True)
            elif args.model == "UnetBIN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True, is_instancenorm=True)
            elif args.model == "UnetBGN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True,is_groupnorm=True)
            elif args.model == "UnetBLN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True,is_layernorm=True)
            elif args.model == "UnetBSN":
                model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                  is_batchnorm=True,is_spectralnorm=True)
            elif args.model == "AttentionUnet":
                model = Unet_Attention(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "AttentionUnetBN":
                model = Unet_Attention_BN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "AttentionUnetBIN":
                model = Unet_Attention_BIN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            # elif args.model == "PSPNet":
            #     model = PSPNet(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            # elif args.model == "PSPNetBN":
            #     model = PSPNet_BN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            # elif args.model == "PSPNetBIN":
            #     model = PSPNet_BIN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "Vgg16":
                model = Vgg16Model(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "Vgg16BN":
                model = Vgg16Model(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                   is_batchnorm=True)
            elif args.model == "Vgg16IN":
                model = Vgg16Model(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                   is_instancenorm=True)
            elif args.model == "Vgg16GN":
                model = Vgg16Model(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                   is_groupnorm=True)
            elif args.model == "Vgg16LN":
                model = Vgg16Model(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                   is_layernorm=True)
            elif args.model == "Vgg16SN":
                model = Vgg16Model(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                                   is_spectralnorm=True)
            elif args.model == "RetinaUnet":
                model = RetinaUnet(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "RetinaUnetBN":
                model = RetinaUnet_BN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "RetinaUnetBIN":
                model = RetinaUnet_BIN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "GAN":
                model = GAN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "GANBN":
                model = GAN_BN(input_shape(args.input_shape, args.input_shape, 3), classes=classes)
            elif args.model == "GANBIN":
                model = GAN_BIN(input_shape(args.input_shape, args.input_shape, 3), classes=classes)

            if args.multigpu != 0:
                model = multi_gpu_model(model, gpus=args.multigpu)

            optimizer = set_optimizer(args.learning_rate)
            metrics = [mean_acc, iou, pre, re, f1, DSC]

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)

            if args.checkpoint:
                model.load_weights(args.checkpoint, True)
                print("Load {} Weight!".format(callback_name))
                print("model=", model)

            # for layer in model.layers:
            #     if layer.name[0:4] == 'conv':
            #         #weight를 가져오니까 weigh시각화도 가능할듯
            #         print(layer.name)
            #         print(layer.weights[0].name)
            #         np.savetxt("./weight/"+layer.name+".csv", np.array(layer.get_weights())[0][0][0], delimiter=",")
            #         weight0Shape = tuple(np.array(layer.get_weights())[0].shape)
            #         print(weight0Shape)
            #         #numpy array로 벡터를 만들어서 weight값에 곱해주면 됨
            #         weight0 = np.ones(weight0Shape, dtype=np.float32) * np.array(layer.get_weights())[0]
            #
            #         weight1 = np.array(layer.get_weights())[1]
            #         layer.set_weights([weight0, weight1])
            #         print(layer.input)
            #         print(layer.output)
            #         print(model)
            #model.summary()
                    #featuremap 시각화
                    #visualize_conv_layer(model, layer.name)



            #callback_list.append()
            model.fit_generator(generator=train_generator,
                                steps_per_epoch=10, verbose=1,
                                epochs=args.epochs, validation_data=val_generator,
                                validation_steps=10, shuffle=True,
                                initial_epoch=args.init_epoch,
                                callbacks=callback_list,
                                use_multiprocessing=use_multiprocessing,
                                workers=workers)

        elif args.method == "adversarial":
            #from sklearn.utils import shuffle

            if args.model == "a1":
                model = Adversarial_1(classes=classes, divide=args.divide_param)
            elif args.model == "a2":
                model = Adversarial_2(classes=classes, divide=args.divide_param)
            elif args.model == "a3":
                model = Adversarial_3(classes=classes, divide=args.divide_param)

            if args.multigpu != 0:
                model = multi_gpu_model(model, gpus=args.multigpu)

            optimizer = set_optimizer(args.learning_rate)
            metrics = [mean_acc, iou, pre, re, f1, DSC]
            if args.loss_fn == "crossentropy":
                loss_g = categorical_crossentropy(l1=args.l1, l2=args.l2)
                loss_d = binary_crossentropy(l1=args.l1, l2=args.l2)
            elif args.loss_fn == "focal":
                loss_g = categorical_focal(l1=args.l1, l2=args.l2)
                loss_d = binary_focal(l1=args.l1, l2=args.l2)

            model['generator'].compile(optimizer=optimizer, loss=loss_g, metrics=metrics)
            model['discriminator'].compile(optimizer=optimizer, loss=loss_d, metrics=metrics)
            model['model'].compile(optimizer=optimizer,
                                   loss={'discriminator_output': loss_d,
                                         'generator_output': loss_g},
                                   metrics={'discriminator_output': ['accuracy', pre, re, f1, DSC],
                                            'generator_output': [mean_acc, iou, pre, re, f1, DSC]})

            if args.checkpoint:
                model['model'].load_weights(args.checkpoint, True)
                print("Load {} Weight!".format(callback_name))

            for epoch in range(args.epochs):
                shuffle(trainset)
                shuffle(valset)
                for step in range(int(len(trainset) / args.batch_size) - 1):
                    trainset_step = trainset[step * args.batch_size:(step):(step + 1) * args.batch_size]
                    valset_step = valset[step * args.batch_size:(step):(step + 1) * args.batch_size]

                    if args.model == "a1":
                        pass
                    elif args.model == "a2":
                        pass
                    elif args.model == "a3":
                        # Train Discriminator
                        model['discriminator'].trainable = True
                        img, mask, label, z_noise = Adversarial_data(args.dataset, trainset_step, args.batch_size // 2,
                                                                     args.input_shape, 'train', classes)
                        generator_img = model['generator'].predict_on_batch([img, mask, z_noise])
                        generator_img = np.argmax(generator_img, axis=-1)
                        generator_img = np.reshape(generator_img, (-1, args.window_size, args.window_size))
                        x_discriminator = np.concatenate((generator_img, mask[:, :, :, 0]), axis=0)
                        y_discriminator = [1] * args.batch_size // 2 + [0] * args.batch_size // 2

                        discriminator_loss = model['discriminator'].train_on_batch(x_discriminator, y_discriminator)

                        # Train Generator
                        model['discriminator'].trainable = False
                        generator_loss = model['model'].train_on_batch(
                            {'img_input': img, 'mask_input': mask, 'z_noise_input': z_noise},
                            {'generator_output': label, 'discriminator_output': [1] * args.batch_size // 2})

            pass

    else: #test
        if args.model == "v3":
            model = DeepLabv3plus(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        elif args.model == "v3BIN":
            model = DeepLabv3plus(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        elif args.model == "AttentionUnet":
            model = Unet_Attention(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        elif args.model == "AttentionUnetBIN":
            model = Unet_Attention_BIN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        # elif args.model == "Unet":
        #     model = Unet(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        # elif args.model == "FCN":
        #     model = FCN(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        elif args.model == "FCN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        elif args.model == "FCNBN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_batchnorm2=True)
        elif args.model == "FCNIN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_instancenorm=True)
        elif args.model == "FCNGN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_groupnorm=True)
        elif args.model == "FCNLN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_layernorm=True)
        elif args.model == "FCNSN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_spectralnorm=True)
        elif args.model == "FCNBBN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_batchnorm=True, is_batchnorm2=True)
        elif args.model == "FCNBIN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_batchnorm=True, is_instancenorm=True)
        elif args.model == "FCNBGN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_batchnorm=True, is_groupnorm=True)
        elif args.model == "FCNBLN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_batchnorm=True, is_layernorm=True)
        elif args.model == "FCNBSN":
            model = FCNModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                             is_batchnorm=True, is_spectralnorm=True)
        elif args.model == "Unet":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes)
        elif args.model == "UnetBN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_batchnorm2=True)
        elif args.model == "UnetIN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_instancenorm=True)
        elif args.model == "UnetGN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_groupnorm=True)
        elif args.model == "UnetLN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_layernorm=True)
        elif args.model == "UnetSN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_spectralnorm=True)
        elif args.model == "UnetBBN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_batchnorm=True, is_batchnorm2=True)
        elif args.model == "UnetBIN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_batchnorm=True, is_instancenorm=True)
        elif args.model == "UnetBGN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_batchnorm=True, is_groupnorm=True)
        elif args.model == "UnetBLN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_batchnorm=True, is_layernorm=True)
        elif args.model == "UnetBSN":
            model = UnetModel(input_shape=(args.input_shape, args.input_shape, 3), classes=classes,
                              is_batchnorm=True, is_spectralnorm=True)
        ##################### featuremap visualization #############################
        feature_dir = './' + args.model + 'Image'
        model.load_weights(args.checkpoint, True)
        print("model=",model.name)
        if not os.path.isdir(feature_dir):
            os.mkdir(feature_dir)
        ixs = []
        names = []
        for i in range(len(model.layers)):
            if model.layers[i].name[0:4] == 'conv': #filtermap 너무 잘게 쪼개지는 부분 출력 x
                if model.layers[i].output.shape[3] == 4096:
                    break
                if model.layers[i].name[0:8] == 'conv2d_t': #transpose 부분 출력 x
                    break
                print(model.layers[i].name)
                print(model.layers[i].output.shape)
                names.append(model.layers[i].name)
                ixs.append(i)
        print(ixs)

        outputs = [model.layers[i].output for i in ixs]
        model = Model(inputs=model.inputs, outputs=outputs)
        # load the image with the required shape
        img = load_img('../dataset/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg', target_size=(512, 512))
        # convert the image to an array
        img = img_to_array(img)
        # expand dimensions so that it represents a single 'sample'
        img = expand_dims(img, axis=0)
        # get feature map for first hidden layer
        feature_maps = model.predict(img)
        #한장씩 저장
        # plot the output from each block
        count = 0
        countWhole = 0
        for fmap in feature_maps:
            for i in range(4):
                pyplot.imshow(fmap[0, :, :, i], cmap='gray')
                pyplot.gcf().set_size_inches(15, 8)
                pyplot.savefig(feature_dir + '/' + str(names[count]) + '_' + str(i) + '.png', format='png')
                countWhole+=1

            count += 1

        # square = 3
        # count = 0
        # countWhole = 0
        # for fmap in feature_maps:
        #     # plot all 64 maps in an 8x8 squares
        #     ix = 1
        #     for _ in range(square):
        #         for _ in range(square):
        #             # specify subplot and turn of axis
        #             ax = pyplot.subplot(square, square, ix)
        #             ax.set_xticks([])
        #             ax.set_yticks([])
        #             # plot filter channel in grayscale
        #             pyplot.imshow(fmap[0, :, :, ix - 1], cmap='gray')
        #             ix += 1
        #         pyplot.savefig(feature_dir + '/' + str(names[count]) + '_' + str(i) + '.png', format='png')
        #         countWhole+=1
        #     count += 1
        print("{}images saved!".format(countWhole))
        ##################### featuremap visualization #############################


        ############################# test ########################################
        import cv2
        #test_steps = 40
        if not os.path.isdir(r'./test_img'):
            os.mkdir(r'./test_img')
        if not os.path.isdir(r'./test_img/{}'.format(args.model)):
            os.mkdir(r'./test_img/{}'.format(args.model))

        model.load_weights(args.checkpoint, True)
        print("model=",model)
        ##Test part needs modify
        #for i in range(test_steps):
        # img = cv2.cvtColor(cv2.imread(r"./dataset/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(valset[i])),
        #                    cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(cv2.imread(r"../dataset/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"),
                           cv2.COLOR_BGR2RGB)
        print("image suceess")
        # lb = cv2.imread(r"./dataset/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassAugRaw/{}.png".format(valset[i]),
        #                 cv2.IMREAD_GRAYSCALE)
        lb = cv2.imread(r"../dataset/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png",
                        cv2.IMREAD_GRAYSCALE)
        print("lb suceess")

        img, lb = scaling(img, lb, 512, 'test')

        # cv2.imwrite("./test_img/{}/{}_test{}_orig.png".format(args.model, args.init_epoch, i),
        #             cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite("./test_img/{}/{}_test2007_000032_orig.png".format(args.model, args.init_epoch),
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # cv2.imwrite("./test_img/{}/{}_test{}_mask.png".format(args.model, args.init_epoch, i), lb * 10)
        cv2.imwrite("./test_img/{}/{}_test2007_000032_mask.png".format(args.model, args.init_epoch), lb * 10)

        img = img[np.newaxis, ...]

        pred = model.predict_on_batch(img)
        pred = np.reshape(pred, (-1, args.input_shape, args.input_shape, classes))
        make_segmap(pred, '2007_000032', args.model, args.init_epoch)


if __name__ == "__main__":
    main()
