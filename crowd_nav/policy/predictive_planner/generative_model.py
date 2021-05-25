"""
Uses an RNN to model the motion of each observed agent.
Based on paper: (ref needed)

Last edits:
08/08/19 - Stuart Eiffert - created
"""

import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn
import copy
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') #if using python3 environment 


class ResponseRNN:
    def __init__(self, MODEL_PATH, stdev=None):

        #define model we will load
        self.cell_type = "LSTM"
        self.seq_length_in = 12
        self.seq_length_out = 1 #for planning. 8 for training
        self.input_dim = 4 #? tbc
        self.output_dim = 5 #updated for use with gaussian output = 5
        self.layers_stacked_count = 2
        self.hidden_dim = 64

        #data format
        self.use_local_map = False #setting this requires changing lots of other dimensions TBD
        self.stdev=stdev #Model trained on normed data, so we must normalise each input. if None, then assumed already normed

        # Backward compatibility for TensorFlow's version 0.12: 
        try:
            tf.nn.seq2seq = tf.contrib.legacy_seq2seq
            tf.nn.rnn_cell = tf.contrib.rnn
            tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
            print("TensorFlow's version : 1.0 (or more)")
        except: 
            print("TensorFlow's version : 0.12")

        tf.reset_default_graph()
        # sess.close()
        self.sess = tf.InteractiveSession()

        #build tf graph
        self.build_model()
        saver = tf.train.Saver()
        saver.restore(self.sess, MODEL_PATH+"/model.ckpt")
        print("Model restored.")

    def build_model(self):
        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            # we do not use the last observed position, as this becomes the tree root
            self.enc_inp = [
                tf.placeholder(tf.float32, shape=(None, self.input_dim), name="inp_{}".format(t))
                for t in range(self.seq_length_in)
            ]

            
            # Decoder: inputs
            self.dec_inp = [
                tf.placeholder(tf.float32, shape=(None, self.input_dim), name="dec_inp_{}".format(t))
                for t in range(self.seq_length_out)
            ]
            
            #For use with LM:
            if self.use_local_map:
                enc_LM = [
                    tf.placeholder(tf.float32, shape=(None, input_dim_LM), name="inp_{}".format(t))
                    for t in range(seq_length_in)
                ]

                # Decoder: LM inputs
                dec_LM = [
                    tf.placeholder(tf.float32, shape=(None, input_dim_LM), name="dec_inp_{}".format(t))
                    for t in range(seq_length_out)
                ]        
           
            # Create a `layers_stacked_count` of stacked RNNs (GRU or LSTM cells here).
            # cell is used for both encoder and decoder here, with non-shared weights
            cells = []
            for i in range(self.layers_stacked_count):
                with tf.variable_scope('RNN_{}'.format(i)):
                    if self.cell_type == "GRU":
                        cells.append(tf.nn.rnn_cell.GRUCell(self.hidden_dim))
                    elif self.cell_type == "LSTM":
                        cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim))
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            if self.use_local_map:
                #num_layers = 2
                w_lm_1 = tf.Variable(tf.random_normal([input_dim_LM, input_dim_LM]))
                b_lm_1 = tf.Variable(tf.random_normal([input_dim_LM], mean=1.0)) 
                w_lm_2 = tf.Variable(tf.random_normal([input_dim_LM, embed_dim_LM])) 
                b_lm_2 = tf.Variable(tf.random_normal([embed_dim_LM], mean=1.0))  
                
                #enc
                enc_lm_1 = [tf.nn.relu(tf.matmul(i, w_lm_1) + b_lm_1) for i in enc_LM]
                enc_lm_2 = [tf.nn.relu(tf.matmul(i, w_lm_2) + b_lm_2) for i in enc_lm_1]
                
                dec_lm_1 = [tf.nn.relu(tf.matmul(i, w_lm_1) + b_lm_1) for i in dec_LM]
                dec_lm_2 = [tf.nn.relu(tf.matmul(i, w_lm_2) + b_lm_2) for i in dec_lm_1]
                
                #enc_inp = tf.concat([enc_inp, enc_lm_2], 1)
                #dec_inp = tf.concat([dec_inp, dec_lm_2], 1)
                enc_inp_concat = []
                dec_inp_concat = []
                for step in range(len(enc_inp)):
                    enc_inp_concat.append(tf.concat([enc_inp[step], enc_lm_2[step]],1))
                    #for agent in range(len(enc_inp[step])):
                    #    enc_inp[step][agent]+=enc_LM[step][agent]#.tolist()
                for step in range(len(dec_inp)):
                    dec_inp_concat.append(tf.concat([dec_inp[step], dec_lm_2[step]],1))
                    #for agent in range(len(dec_inp[step])):
                    #    dec_inp[step][agent]+=dec_LM[step][agent]#.tolist()
                        
                w_in = tf.Variable(tf.random_normal([input_dim+embed_dim_LM, hidden_dim]))

            else:        
                w_in = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim]))
                enc_inp_concat = self.enc_inp
                dec_inp_concat = self.dec_inp
                
            b_in = tf.Variable(tf.random_normal([self.hidden_dim], mean=1.0)) 
            w_out = tf.Variable(tf.random_normal([self.hidden_dim, self.output_dim]))
            b_out = tf.Variable(tf.random_normal([self.output_dim]))

            embed_enc_inp = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in enc_inp_concat]
            embed_dec_inp = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in dec_inp_concat]

            enc_outputs, self.enc_states = self.rnn_encoder(embed_enc_inp, cell)
            dec_outputs, self.dec_states = self.rnn_decoder(embed_dec_inp, self.enc_states, cell)  


            #enc_outputs, self.enc_states = self.rnn_encoder(self.enc_inp, cell)
            #dec_outputs, self.dec_states = self.rnn_decoder(self.dec_inp, self.enc_states, cell)          

            output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
            self.reshaped_outputs = [output_scale_factor * (tf.matmul(i, w_out) + b_out) for i in dec_outputs]

    def rnn_decoder(self, 
                    decoder_inputs,
                    initial_state,
                    cell,
                    loop_function=None,
                    scope=None):
        with tf.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(inp, state)
                outputs.append(output)
        return outputs, state 

    def rnn_encoder(self, 
                    encoder_inputs,
                    cell,
                    dtype=dtypes.float32,
                    scope=None):
        with tf.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            enc_output, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            return enc_output, enc_state

    def encode_observed(self, X, X_lm=None):
        obs_len = len(X)
        if self.stdev is not None:
            X = self.norm_input(np.array(X), self.stdev).tolist()
        feed_dict_enc = {self.enc_inp[t]: X[t] for t in range(obs_len)}
        enc_state = self.sess.run([self.enc_states], feed_dict_enc)
        return enc_state

    def decode_step(self, R, enc_state, R_lm=None):
        if self.stdev is not None:
            R = self.norm_input(np.array([R]),self.stdev).tolist()[0] #R has 1 dimension less than X
        feed_dict_dec = {self.dec_inp[0]: R}
        feed_dict_dec.update({self.enc_states: enc_state}) 
        if R_lm is not None:
            feed_dict_dec.update({dec_LM[t]: R_lm[t] for t in range(len(dec_LM))})
        output, dec_state = self.sess.run([self.reshaped_outputs, self.dec_states], feed_dict_dec)    
        if self.stdev is not None:
            #denorm output:
            output = self.denorm_output(np.array(output),self.stdev).tolist()
        return output, dec_state

    def norm_input(self, data, stdev, mean=0):
        #assumes numpy array for data in shape: (x, x, 4), where 0,2 and 1,3 are same types in last dim
        #if mean !=0:
        #    return None #mean should be zero, otherwise zeros will be normed too (which actually doesnt matter...)
        if stdev[0] == stdev[1]:
            data[:,:,:] = (data[:,:,:]) / stdev[0] #mean should always be 0!
        else:
            data[:,:,0] = (data[:,:,0]) / stdev[0] #mean should always be 0!
            data[:,:,2] = (data[:,:,2]) / stdev[0] 
            data[:,:,1] = (data[:,:,1]) / stdev[1]
            data[:,:,3] = (data[:,:,3]) / stdev[1] 
        return data

    def denorm_output(self, data, stdev, mean=0):
        #assumes numpy array for data in shape: (x, x, 5), where 0,2 and 1,3 are same types in last dim (4 is correlation)
        #denormalising sigma: exp(sigma_de) = exp(sigma)*stdev -> sigma_de = sigma + ln(stdev)
        #if mean !=0:
        #    return None #mean should be zero, otherwise zeros will be normed too (which actually doesnt matter...)
        if stdev[0] == stdev[1]:
            data[:,:,:2] = (data[:,:,:2]) * stdev[0] #mean should always be 0!
            data[:,:,2:4] = (data[:,:,2:4])  + np.log(stdev[0])
        else:
            data[:,:,0] = (data[:,:,0]) * stdev[0] #mean should always be 0!
            data[:,:,2] = (data[:,:,2]) + np.log(stdev[0]) 
            data[:,:,1] = (data[:,:,1]) * stdev[1]
            data[:,:,3] = (data[:,:,3]) + np.log(stdev[1])
        return data

