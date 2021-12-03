import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from spektral.layers import GraphAttention, GraphConvSkip, TAGConv
from tensorflow.keras.initializers import GlorotUniform
from layers import *
import tensorflow_probability as tfp
import numpy as np
from sklearn import metrics
from loss import poisson_loss, NB, ZINB, dist_loss



MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


class SCTAG(tf.keras.Model):

    def __init__(self, X, adj, adj_n, hidden_dim=128, latent_dim=15, dec_dim=None, adj_dim=32):
        super(SCTAG, self).__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
            #dec_dim = [128, 256]
        self.latent_dim = latent_dim
        self.X = X
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.sparse = False

        initializer = GlorotUniform(seed=7)

        # Encoder
        X_input = Input(shape=self.in_dim)
        h = Dropout(0.2)(X_input)
      
        self.sparse = True
        A_in = Input(shape=self.n_sample, sparse=True)
        h = TAGConv(channels=hidden_dim, kernel_initializer=initializer, activation="relu")([h, A_in])
        z_mean = TAGConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])

        self.encoder = Model(inputs=[X_input, A_in], outputs=z_mean, name="encoder")
        clustering_layer = ClusteringLayer(name='clustering')(z_mean)
        self.cluster_model = Model(inputs=[X_input, A_in], outputs=clustering_layer, name="cluster_encoder")

        # Adjacency matrix decoder
        
        dec_in = Input(shape=latent_dim)
        h = Dense(units=adj_dim, activation=None)(dec_in)
        h = Bilinear()(h)
        dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
        self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")
        

        # Expression matrix decoder

        decx_in = Input(shape=latent_dim)
        h = Dense(units=dec_dim[0], activation="relu")(decx_in)
        h = Dense(units=dec_dim[1], activation="relu")(h)
        h = Dense(units=dec_dim[2], activation="relu")(h)

        pi = Dense(units=self.in_dim, activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')(h)

        disp = Dense(units=self.in_dim, activation=DispAct, kernel_initializer='glorot_uniform', name='dispersion')(h)

        mean = Dense(units=self.in_dim, activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')(h)

        # decx_out = Dense(units=self.in_dim)(h)
        self.decoderX = Model(inputs=decx_in, outputs=[pi, disp, mean], name="decoderX")

    def pre_train(self, epochs=1000, info_step=10, lr=1e-4, W_a=0.3, W_x=1, W_d=0, min_dist=0.5, max_dist=20):
      
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.sparse == True:
            self.adj_n = tfp.math.dense_to_sparse(self.adj_n)

        # Training
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, self.adj_n])
                # X_out = self.decoderX(z)
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)

                if W_d:
                    Dist_loss = tf.reduce_mean(dist_loss(z, min_dist, max_dist=max_dist))
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                loss = W_a * A_rec_loss + W_x * zinb_loss
                if W_d:
                    loss += W_d * Dist_loss

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
            if epoch % info_step == 0:
                if W_d:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy(),
                          "  Dist_loss:", Dist_loss.numpy())
                else:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy())

        print("Pre_train Finish!")

    def alt_train(self, y, epochs=300, lr=5e-4, W_a=0.3, W_x=1, W_c=1.5, info_step=8, n_update=8, centers=None):

        self.cluster_model.get_layer(name='clustering').clusters = centers

        # Training
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for epoch in range(0, epochs):

            if epoch % n_update == 0:
                q = self.cluster_model([self.X, self.adj_n])
                p = self.target_distribution(q)

            with tf.GradientTape(persistent=True) as tape:

                z = self.encoder([self.X, self.adj_n])
                q_out = self.cluster_model([self.X, self.adj_n])
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                cluster_loss = tf.reduce_mean(KLD(q_out, p))
                tot_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss

            vars = self.trainable_weights
            grads = tape.gradient(tot_loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            if epoch % info_step == 0:
                print("Epoch", epoch, " zinb_loss: ", zinb_loss.numpy(), " A_rec_loss: ", A_rec_loss.numpy(),
                      " cluster_loss: ", cluster_loss.numpy())

        tf.compat.v1.disable_eager_execution()
        q = tf.constant(q)
        session = tf.compat.v1.Session()
        q = session.run(q)
        self.y_pred = q.argmax(1)

        return self

    def target_distribution(self, q):
        q = q.numpy()
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def embedding(self, count, adj_n):
        if self.sparse:
            adj_n = tfp.math.dense_to_sparse(adj_n)
        return np.array(self.encoder([count, adj_n]))

    def rec_A(self, count, adj_n):
        h = self.encoder([count, adj_n])
        rec_A = self.decoderA(h)
        return np.array(rec_A)

    def get_label(self, count, adj_n):
        if self.sparse:
            adj_n = tfp.math.dense_to_sparse(adj_n)
        clusters = self.cluster_model([count, adj_n]).numpy()
        labels = np.array(clusters.argmax(1))
        return labels.reshape(-1, )
