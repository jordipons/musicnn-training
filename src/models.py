import tensorflow as tf
import models_frontend as frontend
import models_midend as midend
import models_backend as backend
import models_baselines

# disabling deprecation warnings (caused by change from tensorflow 1.x to 2.x)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def model_number(x, is_training, config):

    ############### START BASELINES ###############
    #     dieleman < vgg32 < timbre < vgg128      #

    if config['model_number'] == 0:
        print('\nMODEL: Dieleman | BN input')
        return models_baselines.dieleman(x, is_training, config)
        # 66k params | ROC-AUC: 88.61 | PR-AUC: 34.13 | VAL-COST: 0.1390

    elif config['model_number'] == 1:
        print('\nMODEL: VGG 32 | BN input')
        return models_baselines.vgg(x, is_training, config, 32)
        # 40k params | ROC-AUC: 88.85 | PR-AUC: 34.85 | VAL-COST: 0.1373

    elif config['model_number'] == 2:
        print('\nMODEL: VGG 128 | BN input')
        return models_baselines.vgg(x, is_training, config, 128)
        # 605k params | ROC-AUC: 90.26 | PR-AUC: 38.19 | VAL-COST: 0.1343

    elif config['model_number'] == 3:
        print('\nMODEL: Timbre | BN input')
        return models_baselines.timbre(x, is_training, config, num_filt=1)
        # 185k params | ROC-AUC: 89.28 | PR-AUC: 35.38 | VAL-COST: 0.1368


    ############################## PROPOSED MODELS ##################################
    # rnn < vgg128 < global pooling < global pooling (dense) < autopool < attention #

    elif config['model_number'] == 10:
        print('\nMODEL: BN input > [7, 70%][7, 40%] + temporal > RESIDUAL > GLOBAL POOLING')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=1.6, type='7774timbraltemporal')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = midend_features_list[3] # residual connections: just pick the last of previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='globalpool')
        # 508k params | ROC-AUC: 90.61 | PR-AUC: 38.33 | VAL-COST: 0.1304

    elif config['model_number'] == 11:
        print('\nMODEL: BN input > [7, 70%][7, 40%] + temporal > DENSE > GLOBAL POOLING')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=1.6, type='7774timbraltemporal')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = tf.concat(midend_features_list, 2)  # dense connection: concatenate features from previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='globalpool')
        # 787k params | ROC-AUC: 90.69 | PR-AUC: 38.44 | VAL-COST: 0.1304

    elif config['model_number'] == 12:
        print('\nMODEL: BN input > [7, 40%] > DENSE > ATTENTION + POSITIONAL ENCODING')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=4.5, type='74timbral')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = tf.concat(midend_features_list, 2)  # dense connection: concatenate features from previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='attention_positional')
        # 2.4M params | ROC-AUC: 90.77 | PR-AUC: 38.61 | VAL-COST: 0.1304

    elif config['model_number'] == 13:
        print('\nMODEL: BN input > [7, 40%] > DENSE > AUTOPOOL')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=4.5, type='74timbral')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = tf.concat(midend_features_list, 2)  # dense connection: concatenate features from previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='autopool')
        # 636k params | ROC-AUC: 90.67 | PR-AUC: 38.53 | VAL-COST: 0.1297

    elif config['model_number'] == 14:
        print('\nMODEL: BN input > [7, 70%][7, 40%] + temporal > RESIDUAL > RNN')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=1.6, type='7774timbraltemporal')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = midend_features_list[3] # residual connections: just pick the last of previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='rnn')
        # 3.8M params | ROC-AUC: 90.21 | PR-AUC: 37.17 | VAL-COST: 0.1341

    raise RuntimeError("ERROR: Model {} can't be found!".format(config["model_number"]))






