
from ctypes import *
from typing import List
import csv
import cv2
import numpy as np
import xir
import vart
import os
import math
#import threading
import time
import sys
import queue
#from hashlib import md5
import argparse


DEBUG = False
PRINT_IMAGES = True

BUF_SIZE = 10
imgQ = queue.Queue(BUF_SIZE)
outQ = queue.Queue(BUF_SIZE)

morph_classes = ["0-19", "20-29", "30-39", "40-49", "50-100"]

def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
    ]
    output_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()
    ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

def DEBUG_runDPU(dpu_1):
    print("Start DPU DEBUG with 1 input image")
    # get DPU input/output tensors
    inputTensor_1  = dpu_1.get_input_tensors()
    outputTensor_1 = dpu_1.get_output_tensors()

    input_1_ndim  = tuple(inputTensor_1[0].dims)
    output_1_ndim = tuple(outputTensor_1[0].dims)
    batchSize = input_1_ndim[0]

    out1 = np.zeros([batchSize,1], dtype='int8')

    if DEBUG :
        print(" inputTensor1={}\n".format( inputTensor_1[0]))
        print("outputTensor1={}\n".format(outputTensor_1[0]))

    if not imgQ.empty():
        img_org = imgQ.get()
        # run DPU
        execute_async(
            dpu_1, {
                "quant_input_1": img_org,
                "quant_dense_3_fix": out1
            })

        print("ou1 shape ", out1.shape)
        print("DEBUG DONE")


def runDPU(dpu_1, img):
    # get DPU input/output tensors
    inputTensor_1  = dpu_1.get_input_tensors()
    outputTensor_1 = dpu_1.get_output_tensors()

    input_1_ndim  = tuple(inputTensor_1[0].dims)
    output_1_ndim = tuple(outputTensor_1[0].dims)
    
    batchSize = input_1_ndim[0]

    out1 = np.zeros([batchSize,1], dtype='int8')

    n_of_images = len(img)
    count = 0
    write_index = 0
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_1_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_1_ndim[1:])

        ''' run with batch '''
        # run DPU
        execute_async(
            dpu_1, {
                "quant_input_1": imageRun,
                "quant_dense_3_fix": out1
            })
        cnn_out = out1.copy()
        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = cnn_out[0][j]
            write_index += 1
        count = count + runSize

#def app(images_dir,threads,model_name):
def app(images_dir,model_name):

    images_list=os.listdir(images_dir)
    runTotal = len(images_list)
    print('Found',len(images_list),'images - processing',runTotal,'of them')

    ''' global list that all threads can write results to '''
    global out_q
    out_q = [None] * runTotal

    ''' get a list of subgraphs from the compiled model file '''
    g = xir.Graph.deserialize(model_name)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph0 = subgraphs[0]
    dpu_subgraph1 = subgraphs[1]
    
    if DEBUG:
        print("dpu_subgraph0 = " + dpu_subgraph0.get_name()) #dpu_subgraph0=subgraph_CNN__input_0
        print("dpu_subgraph1 = " + dpu_subgraph1.get_name()) #dpu_subgraph1 = subgraph_CNN__CNN_Conv2d_conv1__18

    dpu_1 = vart.Runner.create_runner(dpu_subgraph1, "run")

    ''' DEBUG with 1 input image '''
    if DEBUG:
        dbg_img = []
        path = "./morph_test/321896_00F16.JPG"
        dbg_img.append(preprocess_fn(path))
        imgQ.put(dbg_img[0])
        DEBUG_runDPU(dpu_1)
        return

    ''' Pre Processing images '''
    print("Pre-processing ",runTotal," images")
    img = []
    for i in range(runTotal):
        path = os.path.join(images_dir,images_list[i])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = np.asarray( [image] )
        img.append(data)

    ''' DPU execution ''' 
    print("run DPU")
    start=0
    end = len(img)
    in_q = img[start:end]
    time1 = time.time()
    runDPU(dpu_1, img)
    time2 = time.time()
    timetotal = time2 - time1
    fps = float(runTotal / timetotal)
    print(" ")
    print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps,runTotal, timetotal))
    print(" ")

    ''' Post Processing '''
    print("Post-processing")
    classes = morph_classes
    correct = 0
    wrong = 0
    file = open('result_age.csv','w',newline='') 
    writer = csv.writer(file)
    writer.writerow(["Path", "Età reale", "Età stimata", "Label stimato"])
    for i in range(len(out_q)):
        name,_ = images_list[i].split('.',1)
        ground_truth = name[-2:]

        if 0 <= int(ground_truth) <= 19:
            label_true = classes[0]
        elif 20 <= int(ground_truth) <= 29:
            label_true = classes[1]
        elif 30 <= int(ground_truth) <= 39:
            label_true = classes[2]
        elif 40 <= int(ground_truth) <= 49:
            label_true = classes[3]
        else:
            label_true = classes[4]

        if PRINT_IMAGES:
            print("Image number ", i, ": ", images_list[i])
            print("Predicted: ", out_q[i], " ground truth: ", ground_truth, "\n")
            writer.writerow([images_list[i], ground_truth, out_q[i], label_true])

        agemin, agemax = label_true.split('-')

        if int(agemin) <= out_q[i] <= int(agemax):
            correct += 1
        else:
            wrong += 1

    accuracy = correct/len(out_q)
    print("Correct: ",correct," Wrong: ",wrong," Accuracy: ", accuracy)
    file.close()
    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--images_dir', type=str, default='./morph_test/', help='Path to folder of images. Default is images')
  ap.add_argument('-m', '--model',      type=str, default='./AgeGen/Age/Age.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')

  args = ap.parse_args()
  print("\n")
  print ('Command line options:')
  print (' --images_dir : ', args.images_dir)
  print (' --model      : ', args.model)
  print("\n")

  app(args.images_dir,args.model)



if __name__ == '__main__':
  main()
