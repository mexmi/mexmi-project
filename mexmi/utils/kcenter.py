"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade,
Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# import tensorflow as tf
import numpy as np

def pairwise_distances(A, B): #800,10; 200;10
    
    na = np.sum(np.square(A), 1) #800,1
    nb = np.sum(np.square(B), 1) #200,1
    
    na = np.reshape(na, [-1, 1])
    nb = np.reshape(nb, [1, -1])
    # print("na: ", na.shape)
    # print("nb:", nb.shape)
    D = np.sqrt(np.maximum(na - 2 * np.matmul(A, np.transpose(B)) + nb, 0.0))#tf.matmul(A,B,False,True) transpose_a=false, transpose_b=true
    return D

class KCenter(object):
    def __init__(self):
    
        self.A = []
        self.B = []

        # D = []#pairwise_distances(self.A, self.B)

        # D_min = np.min(D, axis=1)
        self.D_min_max = [] #np.reduce_max(D_min)
        self.D_min_argmax =[] #np.argmax(D_min)

    def cal_D_min_max(self, A, B):
        self.A = A
        self.B = B
        D = pairwise_distances(self.A, self.B)
        D_min = np.min(D, axis=1)
        self.D_min_max = np.max(D_min) #scalar
        self.D_min_argmax = np.argmax(D_min) #scalar
        return self.D_min_max, self.D_min_argmax