#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
os.system(f"python train_GAN.py")