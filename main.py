##################################################
## Project: CapsuleNetworks
## Script purpose: Main file to act as interface
## Date: 21st April 2018
## Author: Chaitanya Baweja, Imperial College London
##################################################

import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

#in cases where python is unable to detect your files
sys.path.insert(0, './models')

from convolution_network import convolution_network
from capsule_dynamic import capsule_dynamic
from manager import Manager



#============================================================================================

'''
To covert a string into a boolean value
v: string to be converted
Returns bool or error
'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#use a parser to incorporate inputs
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', dest='model', default="capsule_dynamic", help='model type')

#Image/Output setting
#allows for different inputs
parser.add_argument('--input_width', dest='input_width', default=28, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=28, help='input image height')
parser.add_argument('--input_channel', dest='input_channel', default=1, help='input image channel')
parser.add_argument('--output_dim', dest='output_dim', default=10, help='output dim')

#Training Settings
parser.add_argument('--data', dest='data', default='mnist', help='which dataset to use')
parser.add_argument('--root_path', dest='root_path', default='./data/', help='path of dataset')
parser.add_argument('--epochs', dest='epochs', default=250, help='total number of epochs')
parser.add_argument('--batch_size', dest='batch_size', default=64, help='batch size')

parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, help='learning rate of the optimizer')
parser.add_argument('--decay_steps', dest='decay_steps', default=2000, help='decay steps for learning rate')
parser.add_argument('--decay_rate', dest='decay_rate', default=0.96, help='decay_rate to be used with learning rate')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the optimizer')

parser.add_argument('--m_plus', dest='m_plus', default=0.9, help='m_plus')
parser.add_argument('--m_minus', dest='m_minus', default=0.1, help='m_minus')
parser.add_argument('--lambda_val', dest='lambda_val', default=0.5, help='lambda_val')
parser.add_argument('--reg_scale', dest='reg_scale', default=0.0005, help='reg_scale')

#Test Setting
parser.add_argument('--is_train', dest='is_train', default=True, type=str2bool, help='flag to train')
parser.add_argument('--continue_training', dest='continue_training', default=False, type=str2bool, help='flag to continue training')
parser.add_argument('--rotate', dest='rotate', default=False, type=str2bool, help='rotate image flag')
parser.add_argument('--random_pos', dest='random_pos', default=False, type=str2bool, help='randomly place image on 40 x 40 background')

#Extra folders setting
parser.add_argument('--checkpoints_path', dest='checkpoints_path', default='./checkpoints/', help='saved model checkpoint path')
parser.add_argument('--graph_path', dest='graph_path', default='./graphs/', help='tensorboard graph')

#variable that saves all arguments
args = parser.parse_args()

#============================================================================================

def main(_):

    #used for working with gpus and interoperability of code to cases where gpu is not available
    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    #rather than allocating full memory, use as much required
    #run_config.gpu_options.allow_growth = True

    #defining a tensorflow session
    with tf.Session(config=run_config) as sess:

        #print Dataset and Model
        print("Dataset: %s"%args.data)
        print("Model: %s"%args.model)

        #setting up model to use based on arguments
        if args.model == "convolution_network":
            model = convolution_network(args)
        elif args.model == "capsule_dynamic":
            model = capsule_dynamic(args)

        #create graph and checkpoints folder if they don't exist
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)

        #create a subfolder in checkpoints folder
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.model + "/")
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        args.graph_path = os.path.join(args.graph_path, args.model + "/")
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)

        #manager performs all the training/testing
        manager = Manager(args)

        if args.is_train:
            print('Start Training...')
            manager.train(sess, model)
        else:
            print('Start Testing...')
            manager.test(sess, model)

main(args)
