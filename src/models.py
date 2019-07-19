import tensorflow as tf
import models_frontend as frontend
import models_midend as midend
import models_backend as backend
import models_baselines


def model_number(x, is_training, config):

    ############### START BASELINES ###############

    if config['model_number'] == 1:
        print('\nMODEL: Dieleman | BN input')
        return models_baselines.dieleman(x, is_training, config)
        # 66k params | ROC-AUC: 88.18 | PR-AUC: 32.62 | VAL-COST: 0.1399

    elif config['model_number'] == 2:
        print('\nMODEL: Choi small | BN input')
        return models_baselines.choi_small(x, is_training, config)
        # 450k params | ROC-AUC: 89.7 | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 222:
        print('\nMODEL: Choi small | NO DROPOUT | BN input')
        return models_baselines.choi_small(x, is_training, config)
        # 450k params | ROC-AUC: 89.7 | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 3:
        print('\nMODEL: Timbre | BN input')
        return models_baselines.timbre(x, is_training, config, num_filt=1)
        # 185k params | ROC-AUC: 89.57 | PR-AUC: ? | VAL-COST: ?

    ############### PROPOSED MODELS ###############

    elif config['model_number'] == 13:
        print('\nMODEL: BN input > [7, 70%][7, 40%] + temporal > RESIDUAL > GLOBAL POOLING')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=1.6, type='7774timbraltemporal')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = midend_features_list[3] # residual connections: just pick the last of previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='globalpool')
        # 508k params | ROC-AUC: ? | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 17:
        print('\nMODEL: BN input > [7, 70%][7, 40%] + temporal > DENSE > GLOBAL POOLING')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=1.6, type='7774timbraltemporal')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = tf.concat(midend_features_list, 2)  # dense connection: concatenate features from previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='globalpool')
        # 787k params | ROC-AUC: ? | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 53:
        print('\nMODEL: BN input > [7, 40%] > DENSE > ATTENTION + POSITIONAL ENCODING')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=4.5, type='74timbral')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = tf.concat(midend_features_list, 2)  # dense connection: concatenate features from previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='attention_positional')
        # 2.4M params | ROC-AUC: ? | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 23:
        print('\nMODEL: BN input > [7, 40%] > DENSE > AUTOPOOL')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=4.5, type='74timbral')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = tf.concat(midend_features_list, 2)  # dense connection: concatenate features from previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='autopool')
        # 636k params | ROC-AUC: ? | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 36:
        print('\nMODEL: BN input > [7, 40%] > RESIDUAL > RNN')
        frontend_features_list = frontend.musically_motivated_cnns(x, is_training, config['audio_rep']['n_mels'], num_filt=1.6, type='7774timbraltemporal')
        frontend_features = tf.concat(frontend_features_list, 2) # concatnate features coming from the front-end

        midend_features_list = midend.dense_cnns(frontend_features, is_training, 64)
        midend_features = midend_features_list[3] # residual connections: just pick the last of previous layers

        return backend.temporal_pooling(midend_features, is_training, 50, 200, type='rnn')
        # 3.8M params | ROC-AUC: ? | PR-AUC: ? | VAL-COST: ?

    raise RuntimeError("ERROR: Model {} can't be found!".format(config["model_number"]))






