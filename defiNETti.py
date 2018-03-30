"""Train and test neural networks using exchangeable nets and simulation-on-the-fly

If you use this software, please cite:
"A likelihood-free inference framework for population genetic data using exchangeable neural networks"
by Chan, J., Perrone, V., Spence, J.P., Jenkins, P.A., Mathieson, S., and Song, Y.S. 
"""
from __future__ import division
import logging,threading,os
import numpy as np
import tensorflow as tf


def _dimension_match(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for d1,d2 in zip(shape1,shape2):
        if d1 != d2:
            return False
    return True

def _total_dim(dims):
    if type(dims) == int:
        return dims
    to_return = 1
    for d in dims:
        to_return *= d
    return to_return

class _DimensionException(Exception):
    pass


def _layer_to_weights(prev_dim, layer):
    if layer[0] == "fc":
        prev_dim = _total_dim(prev_dim)
        return tf.Variable(tf.truncated_normal((prev_dim,layer[1]),stddev=0.05)), tf.Variable(tf.constant(0.01,shape=[layer[1]])), (layer[1],1)
    if layer[0] == "conv":
        if type(prev_dim) == int:
            conv_shape = (1,layer[1],1,layer[2])
        else:
            conv_shape = (1,layer[1],prev_dim[1],layer[2])
        new_dim = (prev_dim[0] - layer[1] + 1, layer[2])
        return tf.Variable(tf.truncated_normal(conv_shape,stddev=0.05)), tf.Variable(tf.constant(0.01,shape=[layer[2]])), new_dim
    raise Exception("phi and h layers must be either fully connected ('fc',  <#nodes>) or convolutional ('conv', <#width>, <#output depth>)")




def _g_to_dimension(n, layer):
    if layer[0] == "max":
        return 1
    if layer[0] == "moments":
        return len(layer) - 1
    if layer[0] == "sort":
        return n
    if layer[0] == "top_k":
        assert layer[1] <= n
        assert layer[1] >= 1
        return layer[1]
    raise Exception("g layer must be either ('max',), ('sort',), ('top_k', <k>), or ('moments', <m1>, <m2>, ...)")





def _layer(x, layer, weights, bias):
    if layer[0] == "fc":
        flat_shape = x.shape.as_list()
        flat_shape = [-1, flat_shape[1], flat_shape[2]*flat_shape[3]]
        x = tf.reshape(x,shape=flat_shape)
        return tf.expand_dims(tf.nn.relu(tf.tensordot(x, weights,axes=[[2],[0]]) + bias),-1)
    if layer[0] == "conv":
        return tf.nn.relu(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID') + bias)
    if layer[0] == "matmul":
        flat_shape = x.shape.as_list()
        flat_shape = [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]
        x = tf.reshape(x, shape=flat_shape)
        return tf.matmul(x, weights) + bias
    if layer[0] == "softmax":
        flat_shape = x.shape.as_list()
        flat_shape = [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]
        x = tf.reshape(x, shape=flat_shape)
        return tf.nn.log_softmax(tf.matmul(x, weights) + bias)
    raise Exception("An error was encountered while building the network")



def _symmetric_layer(x, layer):
    if layer[0] == "moments":
        fake_x = tf.where(tf.equal(x,0),x+1,x) #set zeros to be one, they will be ignored on the next like
        x = tf.concat([tf.reduce_mean(tf.where(tf.equal(x,0),x,fake_x**m),axis=1,keepdims=True) for m in layer[1:]],axis=1)
        flat_shape = x.shape.as_list()
        flat_shape = [-1, 1, flat_shape[2], flat_shape[1]*flat_shape[3]]
        return tf.reshape(x,shape=flat_shape)

    x = tf.transpose(x,perm=[0,3,2,1])
    if layer[0] == "max":
        k = 1
    elif layer[0] == "sort":
        k = x.shape.as_list()[-1]
    elif layer[0] == "top_k":
        k = layer[1]
    else:
        raise Exception("g layer must be either ('max',), ('sort',), ('top_k', <k>), or ('moments', <m1>, <m2>, ...)")
    if k > 1:
        x = tf.nn.top_k(x, k=k, sorted=True)[0]
    elif k == 1:    
        x = tf.reduce_max(x, 3, keepdims = True)
    else:
        raise Exception("k should not be < 0.")
    flat_shape = x.shape.as_list()
    flat_shape = [-1, 1, flat_shape[2], flat_shape[1]*flat_shape[3]]
    return tf.reshape(x,shape=flat_shape)



def _build_network(phi_net, g, h_net, input_shape, output_shape):

    phi_weights = []
    phi_bias = []
    h_weights = []
    h_bias = []

    #get weights for pre-exchangeable part
    if len(input_shape) == 2:
        prev_dim = (input_shape[1],1)
    else:
        prev_dim = (input_shape[1], input_shape[2])
    for layer in phi_net:
        this_w, this_b, prev_dim = _layer_to_weights(prev_dim, layer)
        phi_weights.append(this_w)
        phi_bias.append(this_b)
    
    #get weights for post-exchangeable part   
    prev_dim = (prev_dim[0], prev_dim[1]*_g_to_dimension(input_shape[0], g))
    for layer in h_net[:-1]:
        this_w, this_b, prev_dim = _layer_to_weights(prev_dim, layer)
        h_weights.append(this_w)
        h_bias.append(this_b)
    if h_net[-1][0] != "matmul" and h_net[-1][0] != "softmax":
        raise Exception("The final layer of the network must either be matmul or softmax")
    h_weights.append(tf.Variable(tf.truncated_normal((_total_dim(prev_dim),output_shape[0]),stddev=0.05)))
    h_bias.append(tf.Variable(tf.constant(0.01,shape=[output_shape[0]])))

    #define function implied by network
    def network_function(x):
        if len(x.shape.as_list()) == 3:
            to_return = tf.expand_dims(x,-1)
        else:
            to_return = x
        for idx,layer in enumerate(phi_net):
            to_return = _layer(to_return, layer, phi_weights[idx], phi_bias[idx])
        to_return = _symmetric_layer(to_return, g)
        for idx,layer in enumerate(h_net):
            to_return = _layer(to_return, layer, h_weights[idx], h_bias[idx])
        return to_return

    return network_function

def train(input_shape, output_shape, simulator, phi_net = [("fc",1024),("fc",1024)], g = ("max",), h_net = [("fc",512),("softmax",)], 
          network_function = None, loss = "cross-ent", accuracy="classification", num_batches = 20000, batch_size=50, 
          queue_capacity=250, verbosity=100, training_threads=1,
          sim_threads=1, save_path=None, training_summary = None, logfile = "."):
    """Train an exchangeable neural network using simulation-on-the-fly

    Args:
        input_shape: a tuple specifying the shape of the data output by <simulator>
        output_shape: a tuple specifying the shape of the label output by <simulator>
        simulator: a function that returns (data, label) tuples, where data is a <input_shape> and label is an <output_shape> shaped numpy array
        phi_net: a list of layers to perform before the symmetric function, see manual for more details
        g: a tuple specifying the exchangeable function to use, see manual for more details
        h_net: a list of layers to perform after the symmetric function, see manual for more details
        network_function: a function of tensorflow operations specifying the neural net (if present ignores phi_net, g, and h_net)
        loss: Either "cross-ent" for cross-entropy loss or "l2" for l2-loss or a user-defined tensorflow function
        accuracy: Either "classification", None for using loss function as accuracy, or a user-defined tensorflow function
        num_batches: number of mini-batches for training
        batch_size: number of training examples used per mini-batch
        queue_capacity: number of training examples to be held in queue
        verbosity: print every k iterations
        training_threads: number of threads used for neural network operations
        sim_threads: number of threads to used to simulate data
        save_path: file base name to save neural network, if None do not save network
        training_summary: A text file containing the <# batch count> <# loss value> <# Accuracy>. The number of batches saved are determined by verbosity. If None, then nothing is saved.
        logfile: Log extra info to logfile. If logfile='.', logs to STDERR.
    Returns:
        None
    """
    if logfile == ".":
        logging.basicConfig(level=logging.INFO)
    elif logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.INFO)
    
    assert len(input_shape) == 2 or len(input_shape) == 3
    assert len(output_shape) == 1 

    #set up a queue that will simulate data on separate threads while the network trains
    #adapted from https://indico.io/blog/tensorflow-data-input-part2-extensions/
    def safe_simulate():
        try:
            result = simulator()
            if not isinstance(result, tuple): raise TypeError("Simulator did not produce a tuple")
            x,y = simulator()
            if not isinstance(x, np.ndarray): raise TypeError("Simulator did not produce a numpy array as input")
            if not isinstance(y, np.ndarray): raise TypeError("Simulator did not produce a numpy array for the label")
            if not _dimension_match(x.shape,input_shape): raise _DimensionException("input")
            if not _dimension_match(y.shape,output_shape): raise _DimensionException("output")
            return simulator()
        except _DimensionException as e:
            logging.warning("Dimension mismatch between simulations and " + str(e) + " size")
        except Exception as e:
            logging.warning("The provided simulator produced data that was not recognized by defiNETti. The exception encountered was " + str(e))
            
    def simulation_iterator():
        while True:
            x,y = safe_simulate()
            yield x,y
    class SimulationRunner(object):
        def __init__(self):
            self.X = tf.placeholder(dtype=tf.float32, shape=input_shape)
            self.Y = tf.placeholder(dtype=tf.float32, shape=output_shape)
            self.queue = tf.FIFOQueue(shapes=[input_shape,output_shape],
                                      dtypes=[tf.float32, tf.float32],
                                      capacity=queue_capacity)
            self.enqueue_op = self.queue.enqueue([self.X,self.Y])
            self.run = True
        def get_inputs(self, batch_size):
            x_batch, y_batch = self.queue.dequeue_many(batch_size)
            return x_batch, y_batch
        def thread_main(self, sess):
            for x,y in simulation_iterator():
                if self.run:
                    sess.run(self.enqueue_op, feed_dict={self.X:x, self.Y:y})
                else:
                    break
        def start_threads(self, sess, n_threads):
            threads = []
            for n in range(n_threads):
                t = threading.Thread(target=self.thread_main, args=(sess,))
                t.daemon = True
                t.start()
                threads.append(t)
            return threads

    #construct network
    simulator_thread = SimulationRunner()
    x_, y_ = simulator_thread.get_inputs(batch_size)
    if network_function is None:
        function = _build_network(phi_net, g, h_net, input_shape, output_shape)
    else:
        function = network_function
    
    #define loss and accuracy
    y_pred = function(x_)
    if loss == "cross-ent":
        loss_func = tf.reduce_mean(-tf.reduce_sum(y_ * (y_pred), axis=[1]))
    elif loss == "l2":
        loss_func = tf.reduce_mean(tf.reduce_sum((y_-y_pred)**2,axis=[1]))
    else:
        loss_func = loss(y_pred, y_)
    if accuracy is None:
        acc_func = loss_func
    elif accuracy == "classification":
        acc_func = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(y_pred,1)),tf.float32))
    else:
        acc_func = accuracy(y_pred, y_)
    train_step = tf.train.AdamOptimizer(.001).minimize(loss_func)

    #save network information for testing purposes
    input_dims = tf.constant(list(input_shape), dtype=tf.float32, name="input_dims")
    x_test = tf.placeholder(dtype=tf.float32, shape=[None] + list(input_shape), name="test_input")
    if loss == "cross-ent":
        test_function = tf.exp(function(x_test), name="learned_function")
    else:
        test_function = tf.identity(function(x_test), name="learned_function")


    saver = tf.train.Saver()    #for saving the weights later

    #run training
    if training_summary is not None:
        summary = np.zeros((int(np.ceil(float(num_batches) / verbosity)),3))
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=training_threads)) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        simulator_thread.start_threads(sess,n_threads=sim_threads)
        op_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for batch_count in range(num_batches):
            _,loss_val,acc = sess.run([train_step,loss_func,acc_func])
            if batch_count % verbosity == 0:
                logging.info('Batch {} complete'.format(batch_count))
                logging.info('Loss value on current batch = {}'.format(loss_val))
                logging.info('Accuracy on current batch = {}'.format(acc))
                if training_summary is not None:
                    summary[int(batch_count / verbosity),:] = [batch_count, loss_val, acc]
   
        simulator_thread.run = False

        #close all threads
        coord.request_stop()
        coord.join(op_threads)
        
        if save_path is not None:
            saver.save(sess, os.path.expanduser(save_path))

    np.savetxt(training_summary, summary)

    return
   



def test(data, model_path, threads = 1):
    """Use a pre-trained neural network to analyze data

    Args:
        data: a list of numpy arrays on which to run the neural network
        model_path: path to the basename where the network is stored, should be same as save_path in train()
        threads: number of threads used for tensorflow operations
    Returns:
        output: a numpy array containing the network output for each input
    """
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=threads)) as sess:
        #Load the trained model
        loader = tf.train.import_meta_graph(os.path.expanduser(model_path) + ".meta")
        loader.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_path)))
        graph = tf.get_default_graph()
        
        #check that dimensions match for all inputs
        input_dims = sess.run(graph.get_tensor_by_name("input_dims:0"))
        for d in data:
            for idx,dim in enumerate(d.shape[1:]):
                if dim != input_dims[idx+1]:
                    raise Exception("Training and Testing dimension differ")

        #run network
        test_input = graph.get_tensor_by_name("test_input:0")
        learned_function = graph.get_tensor_by_name("learned_function:0")        
        outputs = sess.run(learned_function,{test_input:data})

    return outputs
