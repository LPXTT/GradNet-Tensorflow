# Modified by Peixia Li 2019.

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

MOVING_AVERAGE_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'sf_update_ops'

class SiameseNet:
    learningRates = None

    def __init__(self):
        self.learningRates = {}

    def buildTrainNetwork(self, exemplar, instance, opts, isTraining=True):
        isTrainingOp = tf.convert_to_tensor(isTraining, dtype='bool', name='is_training')
        isTrainingOpF = tf.convert_to_tensor(False, dtype='bool', name='is_training')

        with tf.variable_scope('siamese') as scope:
            l2, zFeat5 = self.extract_gra_fea_template(exemplar, opts, isTrainingOp)
            Score = self.response_map_cal(instance, zFeat5, opts, isTrainingOpF)

        return Score, l2

    def buildTrainNetwork1(self, zFeat2Op, gradOp, instance, opts, isTraining=True):
        isTrainingOp = tf.convert_to_tensor(isTraining, dtype='bool', name='is_training')
        isTrainingOpF = tf.convert_to_tensor(False, dtype='bool', name='is_training')
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE) as scope:
            zFeat_, _ = self.template_update_based_grad(zFeat2Op, gradOp, opts, isTrainingOp)
            scope.reuse_variables()
            Score = self.response_map_cal(instance, zFeat_, opts, isTrainingOpF)
        return Score

    def response_map_cal(self, inputs, zfeat5, opts, isTrainingOp):
        # print("Building Siamese branches...")

        with tf.variable_scope('scala1') as scope:
            # print("Building conv1, bn1, relu1, pooling1...")
            name = tf.get_variable_scope().name
            # scope.reuse_variables()
            # outputs = conv1(inputs, 3, 96, 11, 2)
            outputs = self.conv(inputs, 96, 11, 2, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala2') as scope:
            # print("Building conv2, bn2, relu2, pooling2...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 48, 256, 5, 1)
            # scope.reuse_variables()
            outputs = self.conv(outputs, 256, 5, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala3') as scope:
            # print("Building conv3, bn3, relu3...")
            name = tf.get_variable_scope().name
            # scope.reuse_variables()
            # outputs = conv1(outputs, 256, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala4') as scope:
            # scope.reuse_variables()
            # print("Building conv4, bn4, relu4...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 192, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala5') as scope:
            # scope.reuse_variables()
            outputs5 = self.conv(outputs, 256, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            # tf.add_to_collection('x5', outputs)
        with tf.variable_scope('t_scala6') as scope:
            B = int(outputs5.get_shape()[0])
            outputs_ = []
            for ii in range(B):
                if ii > 0:
                    scope.reuse_variables()
                outputs_t = self.conv_f(tf.expand_dims(outputs5[ii],0), tf.expand_dims(zfeat5[0],0), 1, 1, [1.0, 2.0])
                outputs_.append(outputs_t)
            outputs_ = tf.concat(outputs_, 0)

        return outputs_

    def extract_sia_fea_template(self, inputs, opts, isTrainingOp):
        # print("Building Siamese branches...")

        with tf.variable_scope('scala1') as scope:
            # print("Building conv1, bn1, relu1, pooling1...")
            name = tf.get_variable_scope().name
            # scope.reuse_variables()
            outputs = self.conv(inputs, 96, 11, 2, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala2') as scope:
            # print("Building conv2, bn2, relu2, pooling2...")
            name = tf.get_variable_scope().name
            # scope.reuse_variables()
            outputs = self.conv(outputs, 256, 5, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala3') as scope:
            # print("Building conv3, bn3, relu3...")
            name = tf.get_variable_scope().name
            # scope.reuse_variables()
            outputs = self.conv(outputs, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala4') as scope:
            # scope.reuse_variables()
            # print("Building conv4, bn4, relu4...")
            name = tf.get_variable_scope().name
            outputs = self.conv(outputs, 384, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala5') as scope:
            # scope.reuse_variables()
            outputs = self.conv(outputs, 256, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
        return outputs

    def extract_gra_fea_template(self, inputs, opts, isTrainingOp):
        # print("Building Template branches...")
        with tf.variable_scope('scala1_z') as scope:
            # scope.reuse_variables()
            # print("Building conv1, bn1, relu1, pooling1...")
            name = tf.get_variable_scope().name
            # outputs = conv1(inputs, 3, 96, 11, 2)
            outputs = self.conv(inputs, 96, 11, 2, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala2_z') as scope:
            # scope.reuse_variables()
            # print("Building conv2, bn2, relu2, pooling2...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 48, 256, 5, 1)
            outputs = self.conv(outputs, 256, 5, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs2 = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala3_z') as scope:
            # scope.reuse_variables()
            # print("Building conv3, bn3, relu3...")
            name = tf.get_variable_scope().name
            # outputs = conv1(outputs, 256, 384, 3, 1)
            outputs = self.conv(outputs2, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala4_z') as scope:
            # scope.reuse_variables()
            # print("Building conv4, bn4, relu4...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 192, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala5_z') as scope:
            # scope.reuse_variables()
            outputs = self.conv(outputs, 256, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])

        return outputs2, outputs

    def template_update_based_grad(self, zFeat2, grad2, opts, isTrainingOp):

        with tf.variable_scope('t_scala5_zz') as scope:
            # g2 = tf.stop_gradient(g2)
            outputs = self.conv_(grad2*tf.constant(10000, dtype='float32'), 256, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            zFeat2_up = tf.nn.tanh(outputs)+zFeat2

        with tf.variable_scope('scala3_z') as scope:
            scope.reuse_variables()
            # print("Building conv3, bn3, relu3...")
            name = tf.get_variable_scope().name
            # outputs = conv1(outputs, 256, 384, 3, 1)
            outputs = self.conv(zFeat2_up, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala4_z') as scope:
            scope.reuse_variables()
            # print("Building conv4, bn4, relu4...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 192, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala5_z') as scope:
            scope.reuse_variables()
            outputs = self.conv(outputs, 256, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])

        return outputs, zFeat2_up

    def conv(self, inputs, filters, size, stride, groups, lrs, wds, wd, stddev, name=None):
        channels = int(inputs.get_shape()[-1])
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding='VALID')

        with tf.variable_scope('conv'):
            weights = self.getVariable('weights', shape=[size, size, channels / groups, filters], initializer=tf.truncated_normal_initializer(stddev=stddev), weightDecay=wds[0]*wd, dType=tf.float32, trainable=True)
            # tf.get_variable('weights', shape=[size, size, channels/groups, filters], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32) ,
            biases = self.getVariable('biases', shape=[filters, ], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32), weightDecay=wds[1]*wd, dType=tf.float32, trainable=True)
            # tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

        self.learningRates[weights.name] = lrs[0]
        self.learningRates[biases.name] = lrs[1]

        if groups == 1:
            conv = groupConv(inputs, weights)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

            conv = tf.concat(axis=3, values=convGroups)

        if name is not None:
            conv = tf.add(conv, biases, name=name)
        else:
            conv = tf.add(conv, biases)

        # print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d, Groups = %d' % (size, size, stride, filters, channels, groups))

        return conv

    def conv_(self, inputs, filters, size, stride, groups, lrs, wds, wd, stddev, name=None):
        channels = int(inputs.get_shape()[-1])
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding='SAME')

        with tf.variable_scope('conv'):
            weights = self.getVariable('weights', shape=[size, size, channels / groups, filters], initializer=tf.truncated_normal_initializer(stddev=stddev), weightDecay=wds[0]*wd, dType=tf.float32, trainable=True)
            # tf.get_variable('weights', shape=[size, size, channels/groups, filters], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32) ,
            biases = self.getVariable('biases', shape=[filters, ], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32), weightDecay=wds[1]*wd, dType=tf.float32, trainable=True)
            # tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

        self.learningRates[weights.name] = lrs[0]
        self.learningRates[biases.name] = lrs[1]

        if groups == 1:
            conv = groupConv(inputs, weights)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

            conv = tf.concat(axis=3, values=convGroups)

        if name is not None:
            conv = tf.add(conv, biases, name=name)
        else:
            conv = tf.add(conv, biases)

        # print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d, Groups = %d' % (size, size, stride, filters, channels, groups))

        return conv

    def conv_f(self, inputs, weights_, stride, groups, lrs):
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding='VALID')

        with tf.variable_scope('conv'):
            weights = tf.transpose(weights_, [1,2,3,0])

        self.learningRates[weights.name] = lrs[0]

        if groups == 1:
            conv = groupConv(inputs, weights)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

            conv = tf.concat(axis=3, values=convGroups)

        return conv

    def batchNorm(self, x, isTraining):
        shape = x.get_shape()
        paramsShape = shape[-1:]

        axis = list(range(len(shape)-1))

        with tf.variable_scope('bn'):
            beta = self.getVariable('beta', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32))
            self.learningRates[beta.name] = 1.0
            gamma = self.getVariable('gamma', paramsShape, initializer=tf.constant_initializer(value=1, dtype=tf.float32))
            self.learningRates[gamma.name] = 2.0
            movingMean = self.getVariable('moving_mean', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32), trainable=False)
            movingVariance = self.getVariable('moving_variance', paramsShape, initializer=tf.constant_initializer(value=1, dtype=tf.float32), trainable=False)

        mean, variance = tf.nn.moments(x, axis)
        updateMovingMean = moving_averages.assign_moving_average(movingMean, mean, MOVING_AVERAGE_DECAY)
        updateMovingVariance = moving_averages.assign_moving_average(movingVariance, variance, MOVING_AVERAGE_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingMean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingVariance)

        mean, variance = control_flow_ops.cond(isTraining, lambda : (mean, variance), lambda : (movingMean, movingVariance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=0.001)

        return x

    def maxPool(self, inputs, kSize, _stride):
        with tf.variable_scope('poll'):
            output = tf.nn.max_pool(inputs, ksize=[1, kSize, kSize, 1], strides=[1, _stride, _stride, 1], padding='VALID')

        return output

    def loss(self, score, y, weights):
        a = -tf.multiply(score, y)
        b = tf.nn.relu(a)
        loss = b+tf.log(tf.exp(-b)+tf.exp(a-b))
        # loss = tf.log(1+tf.exp(a))
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_mean(tf.multiply(weights, loss))
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss]+regularization)

        return loss

    def getVariable(self, name, shape, initializer, weightDecay = 0.0, dType=tf.float32, trainable = True):
        if weightDecay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
        else:
            regularizer = None

        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dType, regularizer=regularizer, trainable=trainable)
