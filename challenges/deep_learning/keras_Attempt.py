##
## Example Submission Script for Deep Learning Programming challenge.
##

import sys, os
import numpy as np
import tensorflow as tf
from datetime import datetime

#Local methods:
from alexnet import AlexNet
from data_loader import data_loader

sys.path.append('../')
from generator import Generator


class Model(object):
    def __init__(self, mode = 'train'):
        '''
        Model class for you to modify for brick/ball/cylinder deep learning challenge.
        mode = 'train' or 'test' - could be a useful flag if we want to use dataset augmentations. 
        '''
        if mode == 'train' or mode == 'test':
            self.mode = mode
        else:
            print('Mode must be train or test.')

        # Learning params
        self.learning_rate = 1e-3
        self.minibatch_size = 32
        self.num_iterations = 1500
        self.train_test_split = 0.7

        # Network params
        self.keep_rate = 0.5 #Fraction of connections that we keep when performing dropout

        # How many pretrained alexnet layers to we want to load up?
        # This must be a number between 1 and 8. 
        self.num_layers_to_load = 5

        # How often we want to write the tf.summary data to disk and measure performance
        self.display_step = 5
        self.save_step = 100

        # Path for tf.summary.FileWriter and to store model checkpoints
        self.save_dir = "tf_data/sample_model"

        self.input_image_size = (227, 227)
        self.label_indices = {'brick': 0, 'ball': 1, 'cylinder': 2}
        self.labels_ordered = list(self.label_indices)
        self.num_classes = len(self.label_indices)

        self.channel_means = np.array([147.12697, 160.21092, 167.70029])

        #Go ahead and build graph on initialization:
        self.build_graph()

        #Set to true with we create a tf session.
        self.initialized = False


    def build_graph(self):
        '''
        This is where you can experiment with various architectures. 
        '''

        # TF placeholder for graph input and output
        self.X = tf.placeholder(tf.float32, [None, 
                                        self.input_image_size[0], 
                                        self.input_image_size[1], 
                                        3])

        self.y = tf.placeholder(tf.float32, [None, self.num_classes])
        l2_value = 0.1
        self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (5, 5), input_shape= (227,227,3), activation='relu', padding='valid',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_value)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_value)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(8, (2, 2),activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_value)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation=tf.nn.relu),
                tf.keras.layers.Dense(3, activation=tf.nn.softmax)
            ])
        
        self.model.summary()
        
        # self.keep_prob = tf.placeholder(tf.float32)

        # # Build Alexnet portion of graph
        # self.AN = AlexNet(self.X, self.keep_prob, self.num_layers_to_load)

        # #### ------------ Setup Your Graph Here ------------------ ####

        # # We're taking the front of our graph from AlexNet, now let's 
        # # build the rest - this is where we get to be creative!
        # # Grab the tensor from alexnet that we're going to build from
        # # Be sure to change this if you chnage num_layers_to_load
 
        # flattened = tf.reshape(self.AN.pool5, [-1, 6*6*256])

        # layer6 = tf.layers.dense(inputs = flattened, 
		# 					 units = 2048,
        #                      activation = 'relu',
		# 					 name = 'layer6')
        # layer7 = tf.nn.dropout(layer6,0.5) 
        # layer8 = tf.layers.dense(inputs = layer7, 
        #                 units = 512, 
        #                 activation = 'relu',
        #                 name = 'layer8')





        # Your graph should produce an estimate our our labels, yhat.
        # yhat should be of the same dimension as y.
        # self.yhat = ?

        # self.logits = tf.layers.dense(inputs = layer8, units = 3, activation = None, name = 'fc9')
        # self.yhat = tf.nn.softmax(self.logits)



        # # Make sure you inlude the names of all the variables you would like to train here. 
        # # These may include variables from this part of the graph or the alexnet portion. 
        # # You can give your variables whatever name you like by passing in a name into your layers
        # # for example if you setup a layer like this: tf.layers.dense(.... , name = 'my_fc6')
        # # be sure to add 'my_fc6' to the trainable_variable_names list here:
        # self.trainable_variable_names = ['layer6','layer8','fc9']

        #### ---------------- End Graph Setup --------------------- ####

    def train(self):
        '''
        Train model.
        '''

        # Start by building rest of graph that we needed for training.

        #### ------------ Setup Cost Function Here ------------------ ####

        # Your job here is to compute a cost to pass into our AdamOptimizer below. 
        # cost = 

        #Load up data from image files. 
        data = data_loader(label_indices = self.label_indices, 
                   channel_means = self.channel_means, 
                   train_test_split = self.train_test_split,
                   input_image_size = self.input_image_size, 
                   data_path = '../data')
        
        def tfdata_generator(images, labels, is_training, batch_size=128):
            '''Construct a data generator using tf.Dataset'''

            def preprocess_fn(image, label):
                '''A transformation function to preprocess raw data
                into trainable input. '''
                return image, label

            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            if is_training:
                dataset = dataset.shuffle(1000)  # depends on sample size

            # Transform and batch data at the same time
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                preprocess_fn, batch_size,
                num_parallel_batches=4,  # cpu cores
                drop_remainder=True if is_training else False))
            dataset = dataset.repeat()
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

            return dataset

        print('Done1')
        # #Setup minibatch generators
        # G = Generator(data.train.X, data.train.y, minibatch_size = self.minibatch_size)
        # GT = Generator(data.test.X, data.test.y, minibatch_size = self.minibatch_size)
        # G.generate()
        # GT.generate()

        training_set = tfdata_generator(data.train.X, data.train.y, is_training=True, batch_size=self.minibatch_size)
        testing_set  = tfdata_generator(data.test.X, data.test.y, is_training=False, batch_size=self.minibatch_size)

        # print(G.y)
        self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        # call = tf.keras.callbacks.EarlyStopping(monitor='val_loss')
        self.model.fit(training_set.make_one_shot_iterator(), 
                steps_per_epoch=self.minibatch_size,
                epochs=30,
                validation_data=testing_set.make_one_shot_iterator(),
                validation_steps=self.minibatch_size,
                callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15, verbose=1)])
        # self.model.fit_generator(G.generate(),steps_per_epoch=16, epochs=10)
        # cost = tf.losses.softmax_cross_entropy(self.y, self.logits)

        print('Done2')

        #### ------------ End Cost Function Setup  ------------------ ####

        # We only want to train some of the variables in our overall graph:
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] \
                                                    in self.trainable_variable_names]
        
        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            gradients = tf.gradients(cost, var_list)
            gradients = list(zip(gradients, var_list))

            # Create optimizer and apply gradient descent to the trainable variables
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.apply_gradients(grads_and_vars=gradients)

        # Add gradients to summary
        for gradient, var in gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)

        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)

        # Add the loss to summary
        tf.summary.scalar('cross_entropy', cost)

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries together
        merged_summary = tf.summary.merge_all()

        # Initialize the FileWriter
        train_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'train'))
        test_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'test'))

        # Initialize an saver for store model checkpoints
        # We'll save the pretrained alexnet weights and the new weight values that we train:
        var_list_to_save = [v for v in tf.trainable_variables() if v.name.split('/')[0] in \
                                        self.AN.all_variable_names + self.trainable_variable_names]

        saver = tf.train.Saver(var_list = var_list_to_save)


        #Load up data from image files. 
        data = data_loader(label_indices = self.label_indices, 
                   channel_means = self.channel_means, 
                   train_test_split = self.train_test_split,
                   input_image_size = self.input_image_size, 
                   data_path = '../data')

        #Setup minibatch generators
        G = Generator(data.train.X, data.train.y, minibatch_size = self.minibatch_size)
        GT = Generator(data.test.X, data.test.y, minibatch_size = self.minibatch_size)

        #Launch tf session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.initialized = True

        # Add the model graph to TensorBoard
        train_writer.add_graph(self.sess.graph)

        # Load the pretrained weights into the alexnet portion of our graph
        self.AN.load_initial_weights(self.sess)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          self.save_dir))


        #And Train
        for i in range(self.num_iterations):
            G.generate()
            # And run the training op
            self.sess.run(train_op, feed_dict={self.X: G.X, self.y: G.y, self.keep_prob: self.keep_rate})


            # Generate summary with the current batch of data and write to file
            if i % self.display_step == 0:
                s = self.sess.run(merged_summary, feed_dict={self.X: G.X, self.y: G.y, self.keep_prob: 1.0})
                train_writer.add_summary(s, i)

                GT.generate()
                s = self.sess.run(merged_summary, feed_dict={self.X: GT.X, self.y: GT.y, self.keep_prob: 1.0})
                test_writer.add_summary(s, i)

                train_acc = self.sess.run(accuracy, feed_dict={self.X: G.X, self.y: G.y, self.keep_prob: 1.0})
                test_acc = self.sess.run(accuracy, feed_dict={self.X: GT.X, self.y: GT.y, self.keep_prob: 1.0})
                               
                print(i, ' iterations,', str(G.num_epochs), 'epochs, train accuracy = ', train_acc, ', test accuracy = ', test_acc)
                
            if i % self.save_step == 0 and i > 0: 
                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(self.save_dir,
                                               'model_epoch'+str(G.num_epochs)+'.ckpt')
                save_path = saver.save(self.sess, checkpoint_name)
                
                print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                               checkpoint_name))


    def predict(self, X, checkpoint_dir = None):
        '''
        X: Numpy array of dimension [n, 227, 227, 3].
        Note that n here may not be the same as the minibatch_size defined above.

        Returns:
        yhat_numpy: A numpy array of dimension [n x 3] containing the predicted 
        one hot labels for each image passed in X. 
        '''
        if not self.initialized:
            self.restore_from_checkpoint(checkpoint_dir)

        yhat_numpy = self.sess.run(self.yhat, feed_dict = {self.X : X, self.keep_prob: 1.0})

        return yhat_numpy


    def restore_from_checkpoint(self, checkpoint_dir = None):
        '''
        Restore model from most recent checkpoint in save dir.
        '''

        saver = tf.train.Saver()
        self.sess = tf.Session()

        #Load latest checkpont, use savedir if we're not given a checkpoint dir:
        if checkpoint_dir is None:
            checkpoint_name = tf.train.latest_checkpoint(self.save_dir)
        else:
            checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)

        print(checkpoint_name)

        #Resore session
        saver.restore(self.sess, checkpoint_name)
        self.initialized = True
