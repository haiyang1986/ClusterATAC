import argparse
import sys
import numpy as np
import random
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from subprocess import check_output
import h5py
import re
import math
import pandas as pd
from os.path import splitext, basename, isfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import mixture
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, BatchNormalization, Dropout, Activation, merge
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class AE():
    def __init__(self, X_shape, n_components, epochs=60, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):
        encoding_dim = self.n_components
        original_dim = X.shape[1]
        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        z = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        ae = Model(input, output)
        encoder = Model(input, z)
        ae_loss = mse(input, output)
        ae.add_loss(ae_loss)
        ae.compile(optimizer=Adam())
        ae.fit(X, epochs=self.epochs, batch_size=8, verbose=2)
        return encoder.predict(X)


class VAE():
    def __init__(self, X_shape, n_components, epochs=60, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), seed=0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        encoding_dim = self.n_components
        original_dim = X.shape[1]
        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        z_mean = Dense(encoding_dim)(encoded)
        z_log_var = Dense(encoding_dim)(encoded)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        vae = Model(input, output)
        encoder = Model(input, z)
        reconstruction_loss = mse(input, output)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        vae.fit(X, epochs=self.epochs, verbose=2)
        return encoder.predict(X)


class Encoder_GAN():
    def __init__(self, X_shape, n_components, weight=0.001, epochs=100, batch_size=64):
        self.latent_dim = n_components
        optimizer = Adam()
        self.shape = X_shape[1]
        self.disc = self.build_disc()
        self.weight = weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        input = Input(shape=(self.shape,))
        z_mean, z_log_var, z = self.encoder(input)
        output = self.decoder(z)
        validity = self.disc(z)
        self.gan = Model(input, [output, validity])
        self.gan.compile(loss=['mse', 'binary_crossentropy'],
                         loss_weights=[1 - self.weight, self.weight],
                         optimizer=optimizer)

    def build_encoder(self):
        encoding_dim = self.latent_dim

        def sampling(args):
            z_mean, z_log_var = args
            return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(K.shape(z_mean), seed=0)

        X = Input(shape=(self.shape,))
        model = Dense(encoding_dim, kernel_initializer="glorot_normal")(X)
        model = BatchNormalization()(model)
        z_mean = Dense(encoding_dim, kernel_initializer="glorot_normal")(model)
        z_log_var = Dense(encoding_dim, kernel_initializer="glorot_normal")(model)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        return Model([X], [z_mean, z_log_var, z])

    def build_decoder(self):
        X = Input(shape=(self.latent_dim,))
        model = Dense(self.shape, kernel_initializer="glorot_normal")(X)
        model = Model(X, model)
        return model

    def build_disc(self):
        X = Input(shape=(self.latent_dim,))
        dec = Dense(1, activation='sigmoid', kernel_initializer="glorot_normal")(X)
        output = Model(X, dec)
        return output

    def train(self, df, bTrain=True):
        X_train = df.values.astype(float)
        model_path = "./vae.h5"
        log_file = "./run.log"
        fp = open(log_file, 'w')
        if bTrain:
            # GAN
            valid = np.ones((self.batch_size, 1))
            fake = np.zeros((self.batch_size, 1))
            for epoch in range(self.epochs):
                #  Train Discriminator
                idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                data = X_train[idx]
                latent_fake = self.encoder.predict(data)[2]
                latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))
                d_loss_real = self.disc.train_on_batch(latent_real, valid)
                d_loss_fake = self.disc.train_on_batch(latent_fake, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #  Train Encoder_GAN
                g_loss = self.gan.train_on_batch(data, [data, valid])
                print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                    epoch + 1, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))
                # fp.write("%f\t%f\n" % (g_loss[0], d_loss[0]))
                if (abs(d_loss[1] - 0.5) < 0.01 and epoch > self.epochs - 10):
                    break
            fp.close()
            # self.encoder.save(model_path)
        else:
            self.encoder = load_model(model_path)
        mat = self.encoder.predict(X_train)[0]
        return mat


class ClusterATAC(object):
    def __init__(self, model_path='./model/', n_latent_dim=200, weight=1e-3, epochs=60, batch_size=64):
        self.model_path = model_path
        self.score_path = './score/'
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight = weight
        self.n_components = n_latent_dim

    # feature extract
    def feature_gan(self, df_ori, save_file, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_gan(df_ori)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        fea.to_csv(save_file, header=True, index=True, sep='\t')
        print("feature extract finished!")
        return True

    def feature_vae(self, df_ori, save_file, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_vae(df_ori)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        fea.to_csv(save_file, header=True, index=True, sep='\t')
        print("feature extract finished!")
        return True

    def feature_ae(self, df_ori, save_file, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_ae(df_ori)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        fea.to_csv(save_file, header=True, index=True, sep='\t')
        print("feature extract finished!")
        return True

    def cluster(self, df_ori, n_clusters=18):
        X = self.encoder_gan(df_ori)
        model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
        return model.fit_predict(X)

    def impute(self, X):
        X.fillna(X.mean())
        return X

    def encoder_gan(self, df):
        egan = Encoder_GAN(df.shape, self.n_components, self.weight, self.epochs, self.batch_size)
        return egan.train(df)

    def encoder_vae(self, df):
        vae = VAE(df.shape, self.n_components, self.epochs, self.batch_size)
        return vae.train(df)

    def encoder_ae(self, df):
        ae = AE(df.shape, self.n_components, self.epochs, self.batch_size)
        return ae.train(df)

    def tsne(self, X):
        model = TSNE(n_components=2)
        return model.fit_transform(X)

    def pca(self, X):
        fea_model = PCA(self.n_components)
        return fea_model.fit_transform(X)

    def gmm(self, X, n_clusters=18):
        # model = KMeans(n_clusters=n_clusters, random_state=0)
        # labels = model.fit_predict(X)
        model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
        return model.fit_predict(X)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='ClusterATAC v1.0')
    parser.add_argument("-i", dest='file_input', default="./TCGA_ATAC_peak_Log2Counts_dedup_sample.txt",
                        help="file input")
    parser.add_argument("-m", dest='run_mode', default="feature", help="run_mode: feature, cluster")
    parser.add_argument("-k", dest='cluster_num', type=int, default=18, help="cluster num")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument("-p", dest='approach', default="spectral", help="kmeans, spectral, tsne_gmm, tsne")
    parser.add_argument("-e", dest='epochs', type=int, default=30, help="Number of iterations")
    parser.add_argument("-w", dest='disc_weight', type=float, default=1e-3, help="weight")
    parser.add_argument("-d", dest='latent_dim', type=int, default=200, help="latent dim")
    parser.add_argument("-b", dest='batch_size', type=int, default=64, help="latent dim")
    args = parser.parse_args()
    model_path = './model/'
    atac = ClusterATAC(model_path, n_latent_dim=args.latent_dim, weight=args.disc_weight, epochs=args.epochs,
                       batch_size=args.batch_size)

    if args.run_mode == 'feature':
        base_file = splitext(basename(args.file_input))[0]
        fea_save_file = './data/' + base_file + '.fea'
        clinical_save_file = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        df = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
        atac.feature_gan(df, fea_save_file)

    elif args.run_mode == 'cluster':
        base_file = 'all'
        fea_save_file = './fea/' + base_file + '.clusteratac'
        out_file = './results/' + base_file + '.clusteratac' + str(args.cluster_num)
        clinical_save_file = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        if isfile(fea_save_file):
            X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
            df = pd.read_csv(clinical_save_file, header=0, index_col=0, sep='\t')
            df = df.loc[:, ['acronym']]
            labels = atac.gmm(X.values, args.cluster_num)
            # dbi_score = davies_bouldin_score(X.values, labels)
            # print(args.cluster_num, dbi_score)
            X['label'] = labels + 1
            X = X.loc[:, ['label']]
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'runtime':
        method = 'atac_2'
        base_file = splitext(basename(args.file_input))[0]
        fea_compare_file = './' + base_file + '.compare'
        tsne_input = './' + base_file + '.tsne'
        clinical_input_file = './data/TCGA-ATAC_DataS3_PeaksAndClusters_v1.csv'
        out_file = './' + base_file + '.' + method
        n_clusters = 2
        if isfile(args.file_input):
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            time_start = time.time()
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=0)
                labels = model.fit_predict(X.values)
                print(time.time() - time_start)
            elif method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0, n_init=23,
                                           n_jobs=1)
                labels = model.fit_predict(X.values)
                print(time.time() - time_start)
            elif method == 'tsne_gmm':
                mat = TSNE(n_components=2).fit_transform(X)
                model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
                labels = model.fit_predict(mat)
                print(time.time() - time_start)
            elif method == 'pca_gmm':
                mat = PCA(n_components=args.latent_dim).fit_transform(X)
                model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
                labels = model.fit_predict(mat)
                print(time.time() - time_start)
            elif method == 'tsne':
                mat = TSNE(n_components=2).fit_transform(X)
                cmd = "cd ./DensityPeakCluster && python step2_cluster.py && cd .."
                check_output(cmd, shell=True)
                print(time.time() - time_start)
            elif method == 'atac_18':
                atac.cluster(X, n_clusters=18)
                print(time.time() - time_start)
            elif method == 'atac_22':
                labels = atac.cluster(X, n_clusters=22)
                print(time.time() - time_start)
            elif method == 'atac_2':
                labels = atac.cluster(X, n_clusters=2)
                print(time.time() - time_start)
        else:
            print('file does not exist!')

    elif args.run_mode == 'prepare_compare':
        method = args.approach
        base_file = splitext(basename(args.file_input))[0]
        fea_compare_file = './' + base_file + '.compare'
        tsne_input = './' + base_file + '.tsne'
        clinical_input_file = './data/TCGA-ATAC_DataS3_PeaksAndClusters_v1.csv'
        out_file = './' + base_file + '.' + method
        n_clusters = 18
        if isfile(args.file_input):
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=0)
                labels = model.fit_predict(X.values)
            elif method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0, n_init=23)
                labels = model.fit_predict(X.values)
            elif method == 'tsne_gmm':
                df_labels = pd.read_csv(clinical_input_file, header=0, index_col=0, sep=',')
                df_labels['Cluster Assignment Number'] = df_labels['Cluster Assignment Number'].apply(
                    lambda x: int(x.replace("Cluster_", "")))
                df_labels = df_labels[~df_labels.index.duplicated(keep='first')]
                labels = df_labels.loc[X.index.values, 'Cluster Assignment Number']
                t1 = df_labels.loc[X.index.values, 'tSNE_Dimension1']
                t2 = df_labels.loc[X.index.values, 'tSNE_Dimension2']
                X['x'] = t1.values.astype(float)
                X['y'] = t2.values.astype(float)
                X['label'] = labels.values.astype(int)
                mat = X.loc[:, ['x', 'y']].values.astype(float)
                model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
                labels = model.fit_predict(mat)
            elif method == 'tsne':
                df_labels = pd.read_csv(clinical_input_file, header=0, index_col=0, sep=',')
                df_labels['Cluster Assignment Number'] = df_labels['Cluster Assignment Number'].apply(
                    lambda x: int(x.replace("Cluster_", "")))
                df_labels = df_labels[~df_labels.index.duplicated(keep='first')]
                labels = df_labels.loc[X.index.values, 'Cluster Assignment Number']
            X['label'] = labels.astype(int)
            X = X.loc[:, ['label']]
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'tsne':
        base_file = splitext(basename(args.file_input))[0]
        out_file = './' + base_file + '.tsne'
        fea_save_file = './data/' + base_file + '.fea'
        clinical_save_file = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        if isfile(args.file_input):
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            mat = X.values.astype(float)
            df = pd.read_csv(clinical_save_file, header=0, index_col=0, sep='\t')
            df = df.loc[:, ['acronym']]
            labels = atac.tsne(mat)
            print(labels.shape)
            df['x'] = labels[:, 0]
            df['y'] = labels[:, 1]
            df.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'pca':
        base_file = splitext(basename(args.file_input))[0]
        out_file = './fea/' + base_file + '.pca'
        clinical_save_file = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        if isfile(args.file_input):
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            mat = X.values.astype(float)
            df = pd.read_csv(clinical_save_file, header=0, index_col=0, sep='\t')
            df = df.loc[:, ['acronym']]
            labels = atac.pca(mat)
            fea = pd.DataFrame(data=labels, index=X.index,
                               columns=map(lambda x: 'v' + str(x), range(labels.shape[1])))
            print(fea.shape)
            fea.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'preprocess':
        clinical_input = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        file_input = './data/TCGA_ATAC_peak_Log2Counts_dedup_sample'
        base_file = splitext(basename(file_input))[0]
        df_save_file = './' + base_file + '.txt'
        fea_save_file = './data/' + base_file + '.fea'
        # df need to be sorted first
        if not isfile(file_input):
            web_file = "https://atacseq.xenahubs.net/download/TCGA_ATAC_peak_Log2Counts_dedup_sample"
            cmd = "wget %s -O %s 1>/dev/null" % (web_file, file_input)
            check_output(cmd, shell=True)
        df = pd.read_csv(clinical_input, header=0, sep='\t')
        X1 = pd.read_csv(file_input, header=0, index_col=0, sep='\t')
        ids = []
        dic = {}
        for line in list(X1):
            tmps = line.rstrip().split('-')
            txt = '-'.join(tmps[0:-1])
            ids.append(txt)
            dic[txt] = line.rstrip()
        cols_select = []
        cols_new = []
        for sample in list(df['bcr_patient_barcode']):
            cols_select.append(dic[sample])
            cols_new.append(sample)
        X1 = X1.loc[:, cols_select]
        X1.columns = cols_new
        X1.to_csv(df_save_file, header=True, index=True, sep='\t')
        # X1 = X1.T
        # atac.feature_extract(X1, fea_save_file, n_components=200)

    elif args.run_mode == 'all':
        method = args.approach
        clinical_input = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        file_input = './data/TCGA_ATAC_peak_Log2Counts_dedup_sample'
        base_file = splitext(basename(file_input))[0]
        df_save_file = 'all.atac'
        if method == 'clusteratac' or method == 'vae' or method == 'ae':
            fea_save_file = './fea/'  + 'all.' + method
        else:
            fea_save_file = './fea/' + 'all.pca'
        # df need to be sorted first
        if not isfile(fea_save_file):
            if not isfile(df_save_file):
                if not isfile(file_input):
                    web_file = "https://atacseq.xenahubs.net/download/TCGA_ATAC_peak_Log2Counts_dedup_sample"
                    cmd = "wget %s -O %s 1>/dev/null" % (web_file, file_input)
                    check_output(cmd, shell=True)
                else:
                    df = pd.read_csv(clinical_input, header=0, sep='\t')
                    X1 = pd.read_csv(file_input, header=0, index_col=0, sep='\t')
                    ids = []
                    dic = {}
                    for line in list(X1):
                        tmps = line.rstrip().split('-')
                        txt = '-'.join(tmps[0:-1])
                        ids.append(txt)
                        dic[txt] = line.rstrip()
                    cols_select = []
                    cols_new = []
                    for sample in list(df['bcr_patient_barcode']):
                        cols_select.append(dic[sample])
                        cols_new.append(sample)
                    X1 = X1.loc[:, cols_select]
                    X1.columns = cols_new
                    X1.to_csv(df_save_file, header=True, index=True, sep='\t')
            X = pd.read_csv(df_save_file, header=0, index_col=0, sep='\t')
            print("Input file loaded!")
            X = X.T
            if method == 'clusteratac':
                atac.feature_gan(X, fea_save_file)
            elif method == 'vae':
                atac.feature_vae(X, fea_save_file)
            elif method == 'ae':
                atac.feature_ae(X, fea_save_file)
            else:
                mat = X.values.astype(float)
                labels = PCA(n_components=args.latent_dim).fit_transform(mat)
                fea = pd.DataFrame(data=labels, index=X.index,
                                   columns=map(lambda x: 'v' + str(x), range(labels.shape[1])))
                fea.to_csv(fea_save_file, header=True, index=True, sep='\t')
        base_file = splitext(basename(args.file_input))[0]
        if method == 'pca':
            return
        else:
            out_file = './results/' + 'all.' + method
        clinical_save_file = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
        print(X.shape)
        df = pd.read_csv(clinical_save_file, header=0, index_col=0, sep='\t')
        df = df.loc[:, ['acronym']]
        print(df.shape)
        if method == 'kmeans':
            model = KMeans(n_clusters=args.cluster_num, random_state=0)
            labels = model.fit_predict(X.values)
        elif method == 'spectral':
            model = SpectralClustering(n_clusters=args.cluster_num, assign_labels="discretize", random_state=0, n_init=2)
            labels = model.fit_predict(X.values)
        else:
            labels = atac.gmm(X.values, args.cluster_num)
        X['label'] = labels + 1
        X = X.loc[:, ['label']]
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'BRCA':
        method = args.approach
        clinical_input = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        df_save_file = './BRCA.rna'
        base_file = splitext(basename(df_save_file))[0]
        if method == 'clusteratac' or method == 'vae' or method == 'ae':
            fea_save_file = './fea/' + base_file + '.' + method
        else:
            fea_save_file = './fea/' + base_file + '.pca'
        # df need to be sorted first
        if not isfile(fea_save_file):
            X = pd.read_csv(df_save_file, header=0, index_col=0, sep=',')
            print(X.shape)
            X = X.T
            if method == 'clusteratac':
                atac.feature_gan(X, fea_save_file)
            elif method == 'vae':
                atac.feature_vae(X, fea_save_file)
            elif method == 'ae':
                atac.feature_ae(X, fea_save_file)
            else:
                mat = X.values.astype(float)
                labels = PCA(n_components=args.latent_dim).fit_transform(mat)
                fea = pd.DataFrame(data=labels, index=X.index,
                                   columns=map(lambda x: 'v' + str(x), range(labels.shape[1])))
                print(fea.shape)
                fea.to_csv(fea_save_file, header=True, index=True, sep='\t')
        if method == 'pca':
            return
        else:
            out_file = './results/' + base_file + '.' + method
        clinical_save_file = './data/clinical_PANCAN_patient_with_followup.tsv.clinical'
        X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
        if method == 'kmeans':
            model = KMeans(n_clusters=5, random_state=0)
            labels = model.fit_predict(X.values)
        elif method == 'spectral':
            model = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0, n_init=2)
            labels = model.fit_predict(X.values)
        else:
            labels = atac.gmm(X.values, 5)
        print(labels.shape)
        X['label'] = labels + 1
        X = X.loc[:, ['label']]
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'eval':
        base_file = splitext(basename(args.file_input))[0]
        m_list = {}
        clusteratac_18 = './results/' + base_file + '.clusteratac18'
        clusteratac_22 = './results/' + base_file + '.clusteratac22'
        kmeans_file = './data/' + base_file + '.kmeans'
        tsne_file = './data/' + base_file + '.tsne'
        pca_file = './data/' + base_file + '.pca'
        tsne_gmm_file = './data/' + base_file + '.tsne_gmm'
        spectral_file = './data/' + base_file + '.spectral'
        pvalue_file = "./results/survival.pval"
        m_list[clusteratac_18] = 'ClusterATAC_18'
        m_list[clusteratac_22] = 'ClusterATAC_22'
        m_list[kmeans_file] = 'Kmeans'
        m_list[tsne_file] = 't-SNE-density'
        m_list[tsne_gmm_file] = 't-SNE-GMM'
        m_list[spectral_file] = 'Spectral'
        m_list[pca_file] = 'PCA-GMM'
        if isfile(pvalue_file):
            cmd = 'rm -f %s' % (pvalue_file)
            check_output(cmd, shell=True)
        for fkey in m_list.keys():
            if isfile(fkey):
                r_cmd = 'Rscript eval.R -d %s -m %s -o %s' % (fkey, m_list[fkey], pvalue_file)
                os.system(r_cmd)

    elif args.run_mode == 'K_18':
        base_file = splitext(basename(args.file_input))[0]
        fea_save_file = './data/' + base_file + '.fea'
        out_file = './' + base_file + '.out'
        eval_file = './' + base_file + '.eval'
        ref_file = './id.csv.gz'
        if isfile(fea_save_file):
            df_ref = pd.read_csv(ref_file, header=0, index_col=7, sep=',')
            df_ref = df_ref.loc[:, ['Chromosome', 'Start', 'End', 'Linked_Gene', 'Linked_Gene_Type']]
            df_ref = df_ref[df_ref['Linked_Gene_Type'] == 'protein_coding']
            df_ref = df_ref[~df_ref.index.duplicated(keep='first')]
            df_ref = df_ref.loc[:, ['Chromosome', 'Start', 'End', 'Linked_Gene']]
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            X_list = list(X)
            df_label = pd.read_csv(out_file, header=0, index_col=0, sep='\t')
            y = df_label['label'].astype(int)
            top_n = 200
            top_out = 5
            for i in range(22):
                out_file = "./tmp/C%d_out.txt" % (i + 1)
                print(i + 1)
                y_each = np.zeros((y.shape[0]))
                for j in range(y.shape[0]):
                    if y[j] == i:
                        y_each[j] = 1
                model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
                model.fit(X.values, y_each)
                ranks = model.feature_importances_
                i_top = np.argsort(ranks)[::-1][:top_n]
                cols = []
                for i in i_top:
                    if X_list[i] in df_ref.index:
                        cols.append(X_list[i])
                df_select = df_ref.loc[cols, ['Chromosome', 'Start', 'End', 'Linked_Gene']]
                df_out = df_select.head(min(df_select.shape[0], top_out))
                df_out.to_csv(out_file, header=False, index=True, sep='\t')
        else:
            print('file id.csv does not exist!')

    elif args.run_mode == 'K_2':
        base_file = splitext(basename(args.file_input))[0]
        out_file = './' + base_file + '_gene.csv'
        # ref_file = './id.csv.gz'
        ref_file = 'TCGA_ATAC_peak.all.probeMap.gz'
        if isfile(ref_file):
            df_ref = pd.read_csv(ref_file, header=0, index_col=0, sep='\t')
            df_ref = df_ref.loc[:, ['chrom', 'chromStart', 'chromEnd']]
            # df_ref = df_ref[df_ref['Linked_Gene_Type'] == 'protein_coding']
            df_ref = df_ref[~df_ref.index.duplicated(keep='first')]
            # df_ref = df_ref.loc[:, ['Chromosome', 'Start', 'End', 'Linked_Gene']]
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep=',')
            X = X.loc[X['p_value'] < 0.05, ['HR', 'p_value']]
            X = X.sort_values(by='p_value', ascending=True)
            X_list = list(X.index)
            cols = []
            for i in range(len(X_list)):
                if X_list[i] in df_ref.index:
                    cols.append(X_list[i])
            df_select = df_ref.loc[cols, ['chrom', 'chromStart', 'chromEnd']]
            df_select['HR'] = X['HR']
            df_select['p_value'] = X['p_value']
            df_select.to_csv(out_file, header=True, index=False, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'K_2_top100':
        base_file = splitext(basename(args.file_input))[0]
        fea_save_file = './data/' + base_file + '.fea'
        out_file = './' + base_file + '.out'
        x_file = './' + base_file + '.two'
        if isfile(fea_save_file):
            X = pd.read_csv(args.file_input, header=0, index_col=0, sep='\t').T
            X_list = list(X)
            df_label = pd.read_csv(out_file, header=0, index_col=0, sep='\t')
            y = df_label['label'].astype(int)
            top_n = 100
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, oob_score=True)
            model.fit(X.values, y)
            ranks = model.feature_importances_
            i_top = np.argsort(ranks)[::-1][:top_n]
            X = X.iloc[:, i_top]
            print(X.shape)
            X.to_csv(x_file, header=True, index=True, sep='\t')

        else:
            print('file does not exist!')


if __name__ == "__main__":
    main()
