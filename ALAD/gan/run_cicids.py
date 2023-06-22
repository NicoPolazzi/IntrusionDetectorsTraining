import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import gan.cicids_utilities as network
import data.cicids as data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, matthews_corrcoef

GENERATED_ATTACKS=50000
RANDOM_SEED =  146
FREQ_PRINT = 20 # print frequency image tensorboard [20]
STEPS_NUMBER = 500

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "gan/train_logs/cicids/{}/{}/{}".format(weight, method, rd)


def train_and_test(nb_epochs, weight, method, degree, random_seed):
    """ Runs the Bigan on the cicids dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("GAN.train.cicids.{}".format(method))

    # Placeholders
    input_pl = tf.compat.v1.placeholder(tf.float32, shape=data.get_shape_input(), name="input")

    is_training_pl = tf.compat.v1.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train()

    features=print(trainx.shape)
    
    trainx_copy = trainx.copy()
    testx, testy = data.get_test()

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The GAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.generator
    dis = network.discriminator

    # Sample noise from random normal distribution
    random_z = tf.random.normal([batch_size, latent_dim], mean=0.0, stddev=1.0, name='random_z')

    # Generate images with generator
    generator = gen(random_z, is_training=is_training_pl)
    # Pass real and fake images into discriminator separately
    
    real_d, inter_layer_real = dis(input_pl, is_training=is_training_pl)
    fake_d, inter_layer_fake = dis(generator, is_training=is_training_pl, reuse=True)

    with tf.compat.v1.name_scope('loss_functions'):
        # Calculate seperate losses for discriminator with real and fake images
        real_discriminator_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), real_d, scope='real_discriminator_loss')
        fake_discriminator_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.constant(0, shape=[batch_size]), fake_d, scope='fake_discriminator_loss')

        
        # Add discriminator losses
        discriminator_loss = real_discriminator_loss + fake_discriminator_loss
        # Calculate loss for generator by flipping label on discriminator output
        generator_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), fake_d, scope='generator_loss')

    with tf.compat.v1.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        dvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        gvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        update_ops_gen = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope='generator')
        update_ops_dis = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope='discriminator')

        optimizer_dis = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen): # attached op for moving average batch norm
            gen_op = optimizer_gen.minimize(generator_loss, var_list=gvars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(discriminator_loss, var_list=dvars)

        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

    with tf.compat.v1.name_scope('training_summary'):
        with tf.compat.v1.name_scope('dis_summary'):
            tf.compat.v1.summary.scalar('real_discriminator_loss', real_discriminator_loss, ['dis'])
            tf.compat.v1.summary.scalar('fake_discriminator_loss', fake_discriminator_loss, ['dis'])
            tf.compat.v1.summary.scalar('discriminator_loss', discriminator_loss, ['dis'])

        with tf.compat.v1.name_scope('gen_summary'):
            tf.compat.v1.summary.scalar('loss_generator', generator_loss, ['gen'])


        sum_op_dis = tf.compat.v1.summary.merge_all('dis')
        sum_op_gen = tf.compat.v1.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.compat.v1.variable_scope("latent_variable"):
        z_optim = tf.compat.v1.get_variable(name='z_optim', shape= [batch_size, latent_dim], initializer=tf.compat.v1.truncated_normal_initializer())
        reinit_z = z_optim.initializer
    # EMA
    generator_ema = gen(z_optim, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    # Pass real and fake images into discriminator separately
    real_d_ema, inter_layer_real_ema = dis(input_pl, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    fake_d_ema, inter_layer_fake_ema = dis(generator_ema, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)


    with tf.compat.v1.name_scope('error_loss'):
        delta = input_pl - generator_ema
        delta_flat = tf.compat.v1.layers.flatten(delta)
        gen_score = tf.norm(tensor=delta_flat, ord=degree, axis=1, keepdims=False, name='epsilon')

    with tf.compat.v1.variable_scope('Discriminator_loss'):
        if method == "cross-e":
            dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_d_ema), logits=fake_d_ema)

        elif method == "fm":
            fm = inter_layer_real_ema - inter_layer_fake_ema
            fm = tf.compat.v1.layers.flatten(fm)
            dis_score = tf.norm(tensor=fm, ord=degree, axis=1, keepdims=False,
                             name='d_loss')

        dis_score = tf.squeeze(dis_score)

    with tf.compat.v1.variable_scope('Total_loss'):
        loss = (1 - weight) * gen_score + weight * dis_score

    with tf.compat.v1.variable_scope("Test_learning_rate"):
        step = tf.Variable(0, trainable=False)
        boundaries = [300, 400]
        values = [0.01, 0.001, 0.0005]
        learning_rate_invert = tf.compat.v1.train.piecewise_constant(step, boundaries, values)
        reinit_lr = tf.compat.v1.variables_initializer(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                              scope="Test_learning_rate"))

    with tf.compat.v1.name_scope('Test_optimizer'):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_invert).minimize(loss, global_step=step, var_list=[z_optim], name='optimizer')
        reinit_optim = tf.compat.v1.variables_initializer(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                              scope='Test_optimizer'))

    reinit_test_graph_op = [reinit_z, reinit_lr, reinit_optim]

    with tf.compat.v1.name_scope("Scores"):
        list_scores = loss

    logdir = create_logdir(method, weight, random_seed)

    sv = tf.compat.v1.train.Supervisor(logdir=logdir, save_summaries_secs=None,save_model_secs=120)

    logger.info('Start training...')
    
    with sv.managed_session() as sess:

        logger.info('Initialization done')

        writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)

        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr

            begin = time.time()
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen = [0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, ld, sm = sess.run([train_dis_op, discriminator_loss, sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, lg, sm = sess.run([train_gen_op, generator_loss, sum_op_gen], feed_dict=feed_dict)

                train_loss_gen += lg
                writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_dis))

            epoch += 1
#            sess.graph._unsafe_unfinalize() #ADDED

        feed_dict = {input_pl: trainx_copy[0:10],
                    is_training_pl:False,
                    learning_rate:lr}
        
        attacks=np.zeros((GENERATED_ATTACKS, 119))

        for i in (0, GENERATED_ATTACKS-network.batch_size, network.batch_size):
            j=i+network.batch_size
            attacks[i:j]=sess.run(generator, feed_dict=feed_dict)
        
        np.save('/home/polazzi/eGAN/Efficient-GAN-Anomaly-Detection/data/cicids_generated_attacks.npy', attacks)


        logger.warn('Testing evaluation...')
        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling unl dataset
        testy = testy[inds]
        scores = []
        inference_time = []

        # Testing
        for t in range(nr_batches_test):
            logger.info('Testing : batch  %d out of %d' % (t, nr_batches_test))

            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            # invert the gan
            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            for step in range(STEPS_NUMBER):
                _ = sess.run(optimizer, feed_dict=feed_dict)
            scores += sess.run(list_scores, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)
            sess.run(reinit_test_graph_op)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))
        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 119])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch,
                     is_training_pl: False}

        for step in range(STEPS_NUMBER):
            _ = sess.run(optimizer, feed_dict=feed_dict)
        batch_score = sess.run(list_scores,
                           feed_dict=feed_dict).tolist()

        scores += batch_score[:size]
        
        per = np.percentile(scores, 80)

        y_pred = scores.copy()
        y_pred = np.array(y_pred)
        
        inds = (y_pred < per)
        inds_comp = (y_pred >= per)
        
        y_pred[inds] = 0
        y_pred[inds_comp] = 1

#        y_pred[(y_pred!=0) & (y_pred!=1)]=1 
        precision, recall, f1,_ = precision_recall_fscore_support(testy,y_pred,average='binary')

        accuracy = accuracy_score(testy, y_pred)

        mcc = matthews_corrcoef(testy, y_pred)

        tn, fp, fn, tp = confusion_matrix(testy, y_pred).ravel()
        
        print(
            "Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
            % (precision, recall, f1))
        
        print(
            "Testing : ACC = %.4f | MCC = %.4f | tn, fp, fn, tp = %d , %d , %d , %d  "
            % (accuracy, mcc, tn, fp, fn, tp))
        
        PATH='/home/polazzi/datasets/cicids/'
        cicids_competitors=open(PATH+"cicids_competitors.csv", "a")
        cicids_competitors.write('CICIDS, '+ 
                       'ALAD (eGAN) - '+ str(nb_epochs) + ' epochs, ' +
                       'ALAD Approach, '+
                       '0.6--0.4, '+
                       str(trainx.shape[0])+', '+
                       str(0)+', '+
                       str(attacks.shape[0])+', '+
                       str(np.unique(testy, return_counts=True)[1][0])+', '+
                       str(np.unique(testy, return_counts=True)[1][1])+', '+
                       str(testx.shape[1])+', '+
                       ' NO AUGMENTATION, '+
                       '{}, {}, {}, {}, {:3f}, {:3f} \n'.format(tp, tn, fp, fn, accuracy, mcc))       
        
        cicids_competitors.flush()        
def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.compat.v1.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed)
