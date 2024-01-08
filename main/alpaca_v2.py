import tensorflow as tf
import numpy as np
import time
from copy import deepcopy


class ALPaCAModel(tf.keras.Model):
    def __init__(self, config, preprocess=None, f_nom=None, noise=None):
        super(ALPaCAModel, self).__init__()
        self.config = deepcopy(config)
        self.preprocess = preprocess
        self.f_nom = f_nom
        if noise is not None:
            self.sigma_eps = noise
        else:
            self.sigma_eps = self.config['sigma_eps']
       
        # 初始化模型层
        self.model_layers = [tf.keras.layers.Dense(units, activation=config['activation'])
                             for units in config['nn_layers']]
        
        # 初始化其他模型参数
        last_layer = config['nn_layers'][-1]
        self.K = tf.Variable(tf.random.normal([last_layer, config['y_dim']]), dtype=tf.float32, name='K_init')
        self.L_asym = tf.Variable(tf.random.normal([last_layer, last_layer]), dtype=tf.float32, name='L_asym')
        # self.L = tf.matmul(self.L_asym, tf.transpose(self.L_asym))  # \Lambda_0
        # self.x = tf.Variable(tf.zeros(shape=[None,None,self.x_dim]), dtype=tf.float32, name="x")
        # self.y = tf.Variable(tf.zeros(shape=[None,None,self.x_dim]), dtype=tf.float32, name="y")
        # self.context_x = tf.Variable(tf.zeros(shape=[None,None,self.x_dim]), dtype=tf.float32, name="cx")
        # self.context_y = tf.Variable(tf.zeros(shape=[None,None,self.x_dim]), dtype=tf.float32, name="cy")

        self.SigEps = tf.Variable(np.array(self.sigma_eps), dtype=tf.float32, name='SigEps') if isinstance(self.sigma_eps, list) else self.sigma_eps * tf.eye(config['y_dim'])

        self.SigEps = tf.reshape(self.SigEps, (1,1,config['y_dim'],config['y_dim']))
        # self.num_models = tf.shape(self.context_x)[0]
        # self.max_num_context = tf.shape(self.context_x)[1]*tf.ones((self.num_models,), dtype=tf.int32)
        # self.num_context = tf.Variable(self.max_num_context, shape=(None,))

    def call(self, inputs, training=False):
        x = inputs
        if self.preprocess is not None:
            x = self.preprocess(x)
        for layer in self.model_layers:
            x = layer(x)
        return x

    # 在这里添加 batch_blr, compute_pred_and_nll, batch_matmul 等方法
    def batch_blr(self, X, Y, num):
        # X = X[:num, :]
        # Y = Y[:num, :]
        X = tf.cast(X[:num, :], tf.float32)
        Y = tf.cast(Y[:num, :], tf.float32)
        self.L = tf.matmul(self.L_asym, tf.transpose(self.L_asym))
        Ln_inv = tf.linalg.inv(tf.linalg.matrix_transpose(X) @ X + self.L)
        Kn = Ln_inv @ (tf.linalg.matrix_transpose(X) @ Y + self.L @ self.K)
        return tf.cond(num > 0, true_fn=lambda: (Kn, Ln_inv), false_fn=lambda: (self.K, tf.linalg.inv(self.L)))
    
    def compute_pred_and_nll(self, phi, posterior_K, posterior_L_inv, y=None, f_nom_x=0):
        mu_pred = self.batch_matmul(tf.linalg.matrix_transpose(posterior_K), phi) + f_nom_x

        spread_fac = 1 + self.batch_quadform(posterior_L_inv, phi)
       
        Sig_pred = tf.expand_dims(spread_fac, axis=-1) * tf.reshape(self.SigEps, (1, 1, self.config['y_dim'], self.config['y_dim']))

        predictive_nll = None
        
        if y is not None:
            logdet = self.config['y_dim'] * tf.math.log(spread_fac) + tf.linalg.logdet(self.SigEps)
            Sig_pred_inv = tf.linalg.inv(Sig_pred)

            quadf = self.batch_quadform(Sig_pred_inv, y - mu_pred)

            predictive_nll = tf.squeeze(logdet + quadf, axis=-1)

        # self.rmse_1 = tf.reduce_mean( tf.sqrt( tf.reduce_sum( tf.square(mu_pred - self.y)[:,0,:], axis=-1 ) ) )
        # self.mpv_1 = tf.reduce_mean( tf.linalg.det( Sig_pred[:,0,:,:]) )

        return mu_pred, Sig_pred, predictive_nll
    
    def batch_matmul(self, mat, batch_v, name='batch_matmul'):
        with tf.name_scope(name):
            return tf.linalg.matrix_transpose(tf.matmul(mat,tf.linalg.matrix_transpose(batch_v)))

    def batch_quadform(self, A, b):
        A_dims = A.ndim
        b_dims = b.ndim
        b_vec = tf.expand_dims(b, axis=-1)
        if A_dims == b_dims + 1:
            return tf.squeeze(tf.linalg.matrix_transpose(b_vec) @ A @ b_vec, axis=-1)
        elif A_dims == b_dims:
            Ab = tf.expand_dims(tf.transpose(tf.matmul(A, tf.transpose(b, perm=[0, 2, 1])),perm=[0,2,1]), axis=-1)
            return tf.squeeze(tf.matmul(tf.transpose(b_vec, perm=[0, 1, 3, 2]), Ab), axis=-1)
        else:
            raise ValueError('Matrix size of %d is not supported.' % A_dims)
    
    def batch_2d_jacobian(y, x):
    # 获取 y 和 x 的形状信息
        y_dim = y.shape[-1]
        x_dim = x.shape[-1]
        batched_y = tf.reshape(y, [-1, y_dim])
        batched_x = tf.reshape(x, [-1, x_dim])

    # 使用 GradientTape 计算雅可比矩阵
        jacobians = []
        for i in range(y_dim):
            with tf.GradientTape() as tape:
                tape.watch(batched_x)
                y_i = batched_y[:, i]
            jacobian_i = tape.batch_jacobian(y_i, batched_x)
            jacobians.append(jacobian_i)

        batched_dydx = tf.stack(jacobians, axis=2)

    # 重新整理得到的雅可比矩阵形状
        dydx = tf.reshape(batched_dydx, tf.concat([tf.shape(y)[:-1], [y_dim, x_dim]], axis=0))
        return dydx
    
    

class ALPaCA:
    def __init__(self, config, preprocess=None, f_nom=None, noise=None):
        self.config = deepcopy(config)
        self.model = ALPaCAModel(config, preprocess, f_nom, noise)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['lr'])
        self.summary_writer = tf.summary.create_file_writer('summaries/'+str(time.time()))

    def train(self, dataset, num_train_updates):
        batch_size = self.config['meta_batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']


        with self.summary_writer.as_default():
            for i in range(num_train_updates):
                x, y = dataset.sample(n_funcs=batch_size, n_samples=horizon+test_horizon)
                x = tf.cast(x, tf.float32)
                y = tf.cast(y, tf.float32)
                self.context_y = y[:,:horizon,:]
                self.context_x = x[:,:horizon,:]
                self.y = y[:,horizon:,:]
                self.x = x[:,horizon:,:]
                self.num_context = np.random.randint(horizon+1, size=batch_size)
                


                with tf.GradientTape() as tape:

                    self.phi = self.model(self.x, training=True)
                    self.context_phi = self.model(self.context_x, training=True)
                    self.f_nom_cx = tf.zeros_like(self.context_y)
                    self.f_nom_x = 0

                    if self.model.f_nom is not None:
                        self.f_nom_cx = self.model.f_nom(self.context_x)
                        self.f_nom_x = self.model.f_nom(self.x)
                    
                    self.context_y_blr = self.context_y - self.f_nom_cx
                  

                    self.posterior_K, self.posterior_L_inv = tf.map_fn( lambda x: self.model.batch_blr(*x),
                                                                    elems=(self.context_phi, self.context_y_blr, self.num_context),
                                                                    dtype=(tf.float32, tf.float32) )

                    self.mu_pred, self.Sig_pred, self.predictive_nll = self.model.compute_pred_and_nll(self.phi, self.posterior_K, self.posterior_L_inv, self.y, self.f_nom_x)

                    self.loss = tf.reduce_mean(self.predictive_nll)

                gradients = tape.gradient(self.loss, self.model.trainable_variables + [self.model.K, self.model.L_asym])

                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables + [self.model.K, self.model.L_asym]))


                if i % 50 == 0:
                    tf.summary.scalar('loss', self.loss, step=i)
                    print(f'Update {i}, Loss: {self.loss.numpy()}')
     
                self.summary_writer.flush()

    def test(self, x_c, y_c, x):

        context_phi = self.model.call(x_c, training=False)
        phi = self.model.call(x, training=False)
        
        f_nom_cx = tf.zeros_like(y_c)
        f_nom_x = 0

        if self.model.f_nom is not None:
            f_nom_cx = self.model.f_nom(x_c)
            f_nom_x = self.model.f_nom(x)
                    
        y_c_blr = y_c - f_nom_cx
                  
        # 计算后验参数
        num_models = tf.shape(x_c)[0]
        num_context = tf.shape(x_c)[1]*tf.ones((num_models,), dtype=tf.int32)
        num_context = num_context.numpy()

        #posterior_K, posterior_L_inv = self.model.batch_blr(context_phi, y_c_blr, num_context)
        posterior_K, posterior_L_inv = tf.map_fn( lambda x: self.model.batch_blr(*x),
                                                                    elems=(context_phi, y_c_blr, num_context),
                                                                    dtype=(tf.float32, tf.float32) )
        # 计算预测分布的均值和方差
        mu_pred, Sig_pred, _ = self.model.compute_pred_and_nll(phi, posterior_K, posterior_L_inv)

        return mu_pred, Sig_pred
        
    def save(self, model_path):
        self.model.save_weights(model_path)

    def load(self, model_path):
        self.model.load_weights(model_path)
        

