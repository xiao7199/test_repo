import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob, os

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
import subprocess
import librosa
import librosa.display
from scipy.io.wavfile import write

gan_folder = '/home/yuanxin/code/bitplanes-tracking/gan_input/'
audio_folder = '/home/yuanxin/Downloads/viola_data/audio/'
file_name = ['3229','3230','3231','3232','3233','3234','3235','3236','3237']

def create_data_generator(V,A,batch_size = 100):
    batch_counter = 0
    visual_batch = [None]*batch_size
    audio_batch = [None]*batch_size

    while True:
        train_index = range(V.shape[0])
        np.random.shuffle(train_index)
        for i in range(len(train_index)):
            visual_batch[batch_counter] = np.expand_dims(V[train_index[batch_counter]],axis = 0)
            audio_batch[batch_counter] =  np.expand_dims(A[train_index[batch_counter]],axis = 0)
            batch_counter += 1
            if batch_counter == batch_size:
                yield np.concatenate(visual_batch),np.concatenate(audio_batch)
                batch_counter = 0
                visual_batch = [None]*batch_size
                audio_batch = [None]*batch_size
                

visual_feat_all = []
audio_feat_all = []
xxx = 0
for i in file_name:
    if xxx ==1:
        break
    xxx += 1
    visual_feat = np.load(gan_folder + 'label_' + i + '.npy')
    fname = audio_folder + "GOPR" + i + '.MP4.wav'
    sr = 16000
    audio = utils.load_audio(fname, sample_length=-1, sr=sr)
    sample_length = audio.shape[0]
    spec = utils.specgram(audio,
                n_fft=512,
                hop_length=None,
                mask=True,
                log_mag=True,
                re_im=False,
                dphase=True,
                mag_only=False)
    mag = spec[:,:,0]
    dphase = spec[:,:,1]
    combine = np.concatenate((dphase, mag), axis=0)
    combineT = np.transpose(combine)

    visual_feat = visual_feat.reshape(visual_feat.shape[0],visual_feat.shape[2])
    print visual_feat.shape
    visual_feat_all.append(visual_feat)

    audio_feat = []
    cnt = 0
    bias = 0
    for i in visual_feat:
        if cnt % 40  == 0:
            bias += 1
        audio_feat.append(combineT[cnt + bias])
        cnt += 1
    audio_feat_all.append(audio_feat)

visual_feat_all = np.array(visual_feat_all)
audio_feat_all = np.array(audio_feat_all)
visual_feat_all = np.concatenate(visual_feat_all)
audio_feat_all = np.concatenate(audio_feat_all)

data_gen = create_data_generator(visual_feat_all,audio_feat_all,128)

# import pdb
# pdb.set_trace()



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 514])
# X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([514, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


FX = tf.placeholder(tf.float32, shape=[None, 16])
# X = tf.placeholder(tf.float32, shape=[None, 784])

FD_W1 = tf.Variable(xavier_init([16, 1]))
FD_b1 = tf.Variable(tf.zeros(shape=[1]))

theta_FD = [FD_W1, FD_b1]


Z = tf.placeholder(tf.float32, shape=[None, 16])
# Z = tf.placeholder(tf.float32, shape=[None, 100])
G_W1 = tf.Variable(xavier_init([16, 32]))
G_b1 = tf.Variable(tf.zeros(shape=[32]))

G_W2 = tf.Variable(xavier_init([32, 64]))
G_b2 = tf.Variable(tf.zeros(shape=[64]))

G_W3 = tf.Variable(xavier_init([64, 128]))
G_b3 = tf.Variable(tf.zeros(shape=[128]))

G_W4 = tf.Variable(xavier_init([128, 256]))
G_b4 = tf.Variable(tf.zeros(shape=[256]))

G_W5 = tf.Variable(xavier_init([256, 514]))
G_b5 = tf.Variable(tf.zeros(shape=[514]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]

FZ = tf.placeholder(tf.float32, shape=[None, 514])
# Z = tf.placeholder(tf.float32, shape=[None, 100])
FG_W1 = tf.Variable(xavier_init([514, 256]))
FG_b1 = tf.Variable(tf.zeros(shape=[256]))

FG_W2 = tf.Variable(xavier_init([256, 128]))
FG_b2 = tf.Variable(tf.zeros(shape=[128]))

FG_W3 = tf.Variable(xavier_init([128, 64]))
FG_b3 = tf.Variable(tf.zeros(shape=[64]))

FG_W4 = tf.Variable(xavier_init([64, 32]))
FG_b4 = tf.Variable(tf.zeros(shape=[32]))

FG_W5 = tf.Variable(xavier_init([32, 16]))
FG_b5 = tf.Variable(tf.zeros(shape=[16]))

theta_FG = [FG_W1, FG_W2, FG_W3, FG_W4, FG_W5, FG_b1, FG_b2, FG_b3, FG_b4, FG_b5]



# def sample_Z(m, n):
    # return np.random.uniform(-1., 1., size=[m, n])

def sample_Z(feat_name):
    gan_folder = '/home/yuanxin/code/bitplanes-tracking/gan_input/'
    visual_feat = np.load(gan_folder + 'label_' + feat_name + '.npy')
    visual_feat = visual_feat.reshape(visual_feat.shape[0],visual_feat.shape[2])
    return visual_feat

def sample_X(feat_name):
    gan_folder = '/home/yuanxin/code/bitplanes-tracking/gan_input/'
    visual_feat = np.load(gan_folder + 'label_' + feat_name + '.npy')
    visual_feat = visual_feat.reshape(visual_feat.shape[0],visual_feat.shape[2])
    return visual_feat

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    G_log_prob = tf.matmul(G_h4, G_W5) + G_b5
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def F_generator(z):
    FG_h1 = tf.nn.relu(tf.matmul(z, FG_W1) + FG_b1)
    FG_h2 = tf.nn.relu(tf.matmul(FG_h1, FG_W2) + FG_b2)
    FG_h3 = tf.nn.relu(tf.matmul(FG_h2, FG_W3) + FG_b3)
    FG_h4 = tf.nn.relu(tf.matmul(FG_h3, FG_W4) + FG_b4)
    FG_log_prob = tf.matmul(FG_h4, FG_W5) + FG_b5
    FG_prob = tf.nn.sigmoid(FG_log_prob)

    return FG_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

def F_discriminator(x):
    # FD_h1 = tf.nn.relu(tf.matmul(x, FD_W1) + FD_b1)
    FD_logit = tf.matmul(x, FD_W1) + FD_b1
    FD_prob = tf.nn.sigmoid(FD_logit)

    return FD_prob, FD_logit

def concat_to_spec(combine): 
    XX = np.array(np.vsplit(combine, 2))
    p = XX[0]
    m = XX[1]
    p = p[:, :, np.newaxis]
    m = m[:, :, np.newaxis]
    XXX = np.concatenate((m,p),axis = 2)
    print XXX.shape
    return XXX

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

FG_sample = F_generator(FZ)
FD_real, FD_logit_real = F_discriminator(FX)
FD_fake, FD_logit_fake = F_discriminator(FG_sample)


# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
print tf.shape(G_loss_1)
G_loss_2 = tf.reduce_mean(tf.nn.l2_loss(G_sample - X))
print tf.shape(G_loss_2)
l = 0.01
G_loss = G_loss_1 + l * G_loss_2

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

FD_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=FD_logit_real, labels=tf.ones_like(FD_logit_real)))
FD_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=FD_logit_fake, labels=tf.zeros_like(FD_logit_fake)))
FD_loss = FD_loss_real + FD_loss_fake
FG_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=FD_logit_fake, labels=tf.ones_like(FD_logit_fake)))
print tf.shape(FG_loss_1)
FG_loss_2 = tf.reduce_mean(tf.nn.l2_loss(FG_sample - FX))
print tf.shape(FG_loss_2)
l = 0.01
FG_loss = FG_loss_1 + l * FG_loss_2

FD_solver = tf.train.AdamOptimizer().minimize(FD_loss, var_list=theta_FD)
FG_solver = tf.train.AdamOptimizer().minimize(FG_loss, var_list=theta_FG)

mb_size = 128
Z_dim = 100

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
D_loss_curr = np.inf
G_loss_curr = np.inf
FD_loss_curr = np.inf
FG_loss_curr = np.inf

reconstruction = False

for it in range(1000000):
    # if it % 1000 == 0:
    #     # samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
    #     samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
    #     # fig = plot(samples)
    #     # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    #     i += 1
    #     # plt.close(fig)
    # every 1000 iter test
   
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z('3230')})
        samples = np.transpose(samples)
        XXX = concat_to_spec(samples)
        ori = utils.ispecgram(XXX,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=False,
              num_iters=1000)
        write('out/{}.wav'.format(str(i).zfill(3)), 16000, ori)
        i += 1

        # print samples.shape
        # fig = plot(samples)
        # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        # i += 1
        # plt.close(fig)

    V_mb, A_mb = data_gen.next()
    # print 'V_mb', V_mb
    # print 'A_mb', A_mb
    # if (D_loss_curr) > 0.1:
    # _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: A_mb, Z: V_mb})
    # _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: A_mb, Z: V_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: A_mb, Z: V_mb})

    if(reconstruction):
        _, FD_loss_curr = sess.run([FD_solver, FD_loss], feed_dict={FX: V_mb, FZ: A_mb})
    # _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
        _, FG_loss_curr = sess.run([FG_solver, FG_loss], feed_dict={FX: V_mb, FZ: A_mb})
        _, FG_loss_curr = sess.run([FG_solver, FG_loss], feed_dict={FX: V_mb, FZ: A_mb})
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        if(reconstruction):
            print('RD loss: {:.4}'. format(FD_loss_curr))
            print('RG_loss: {:.4}'.format(FG_loss_curr))