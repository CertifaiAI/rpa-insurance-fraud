'''
Created on Sep 13, 2019

@author: LEE CHUN WEI
'''
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import sys

temlist = list(myinput.split(sep=","))
np_input = np.array(temlist, dtype=float)
print (np_input)
listout = ''

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(model_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
 
            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="", op_dict=None, producer_op_list=None)
 
            for op in graph.get_operations():
                    print ("Operation Name :",op.name)        
                    print ("Tensor Stats :",str(op.values()))     
                     
        l_input = graph.get_tensor_by_name('x:0') # Input Tensor
        l_output = graph.get_tensor_by_name('out:0') # Output Tensor
         
        print ("Shape of input : ", tf.shape(l_input))
        tf.compat.v1.global_variables_initializer()
         
        Inference_Out = sess.run(l_output, feed_dict = {l_input : np_input.reshape(1,32)})
        print(Inference_Out)
        listout = Inference_Out.tolist()
temr = 0
temcon = 0.0
if listout[0][0] > listout[0][1]:
    temr = 0
    temcon = listout[0][0]
else:
    temr = 1
    temcon = listout[0][1]
PredictedClass = str(temr)
Confidence = str(temcon)
output = str(listout[0][0])+ "," + str(listout[0][1])
print(type(output))