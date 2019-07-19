import tensorflow as tf
import models_baselines
import models_musically_motivated
import models_temporal_pool
import models_transformer

def model_number(x, is_training, config):

    ############### BASELINES ###############

    if config['model_number'] == 0:
        print('\nMODEL: Dieleman | BN input')
        print('-----------------------------------\n')
        return models_baselines.dieleman(x, is_training, config)
        # 66k params | ROC-AUC: 88.18 | PR-AUC: 32.62 | VAL-COST: 0.1399

    elif config['model_number'] == 6969696:
        import class_upc
        out_frontend = class_upc.frontend(x, is_training, num_filt=8)
        return class_upc.backend(out_frontend, is_training, num_filt=64, output_units=200)

    elif config['model_number'] == 1:
        print('\nMODEL: Choi big | BN input')
        print('-----------------------------------\n')
        return models_baselines.choi_big(x, is_training, config)
        # ? params | ROC-AUC: ? | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 2:
        print('\nMODEL: Choi small | BN input')
        print('-----------------------------------\n')
        return models_baselines.choi_small(x, is_training, config)
        # 450k params | ROC-AUC: 89.7 | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 3:
        print('\nMODEL: Timbre | BN input')
        print('-----------------------------------\n')
        return models_baselines.timbre(x, is_training, config, num_filt=1)
        # 185k params | ROC-AUC: 89.57 | PR-AUC: ? | VAL-COST: ?

    elif config['model_number'] == 4: ############################### THE ORIGINAL FROM ISMIR ################################
        print('\n ISMIR musically motivated | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.ismir_frontend(x, is_training, config, num_filt=8, type='ismir_original')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    ############### MEAN/MAX POOL ###############

    elif config['model_number'] == 10:
        print('\n[7, 70%] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=3.5, type='77timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    elif config['model_number'] == 11:
        print('\n[7, 40%] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    elif config['model_number'] == 12:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    elif config['model_number'] == 13: ################################# GOOD ONE #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    elif config['model_number'] == 133: ################################# GOOD ONE BIG #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | BIG (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.2, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=128, output_units=200, type='globalpool')

    elif config['model_number'] == 1312: ################################# GOOD ONE MP12 #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_mp12')

    elif config['model_number'] == 1323: ################################# GOOD ONE MP23 #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_mp23')

    elif config['model_number'] == 13123: ################################# GOOD ONE MP123 #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_mp23_mp12')

    elif config['model_number'] == 1344: ################################# GOOD ONE MP44 #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_mp441_mp442')


    elif config['model_number'] == 14:
        print('\n[7, 70%][7, 40%] [1/3 bands temporal] | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporalthird')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    elif config['model_number'] == 15:
        print('\n[7, 70%][7, 40%] | DENSE | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_dense')

    elif config['model_number'] == 16:
        print('\n[7, 40%] | DENSE | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_dense')

    elif config['model_number'] == 17: ################################# GOOD ONE #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | DENSE | SMALL (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool_dense')

    elif config['model_number'] == 177: ################################# GOOD ONE BIG #################################
        print('\n[7, 70%][7, 40%] [all bands temporal] | DENSE | BIG (for MTT)')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=128, output_units=200, type='globalpool_dense')


    ############### AUTO POOL ###############

    #elif config['model_number'] == 20:
    #    print('\n[7, 70%][7, 40%] [1/3 bands temporal] | SMALL (for MTT) w/ attention and positional embedding')
    #    print('-----------------------------------\n')
    #    out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1, type='7774timbraltemporalthird')
    #    return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool')

    elif config['model_number'] == 20:
        print('\n[7, 40%] | SMALL (for MTT) w/ autopool')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool')

    elif config['model_number'] == 21:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ autopool')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool')

    elif config['model_number'] == 22:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ autopool')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool')

    elif config['model_number'] == 23: ################################# GOOD ONE #################################
        print('\n[7, 40%] | SMALL (for MTT) w/ autopool + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_dense')

    elif config['model_number'] == 233: ################################# GOOD ONE BIG #################################
        print('\n[7, 40%] | BIG (for MTT) w/ autopool + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=7, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=200, output_units=200, type='autopool_dense')

    elif config['model_number'] == 24:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ autopool + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_dense')

    elif config['model_number'] == 25:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ autopool + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_dense')


    ############### AUTO POOL + POSITION EMBEDDING ###############

    elif config['model_number'] == 60:
        print('\n[7, 40%] | SMALL (for MTT) w/ autopool + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_position_sin')

    elif config['model_number'] == 61:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ autopool + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_position_sin')

    elif config['model_number'] == 62:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ autopool + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_position_sin')

    elif config['model_number'] == 63:
        print('\n[7, 40%] | SMALL (for MTT) w/ autopool + dense + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_position_sin_dense')

    elif config['model_number'] == 64:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ autopool + dense + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_position_sin_dense')

    elif config['model_number'] == 65:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ autopool + dense + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='autopool_position_sin_dense')

    ############### RNN ###############

    elif config['model_number'] == 34:
        print('\n[7, 40%] | SMALL (for MTT) w/ rnn x512')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='rnn512')

    elif config['model_number'] == 35:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ rnn x512')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='rnn512')

    elif config['model_number'] == 36:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ rnn x512')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='rnn512')

    elif config['model_number'] == 37:
        print('\n[7, 40%] + DENSE | SMALL (for MTT) w/ rnn x512')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='rnn512_dense')

    elif config['model_number'] == 38:
        print('\n[7, 70%][7, 40%] + DENSE | SMALL (for MTT) w/ rnn x512')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='rnn512_dense')

    elif config['model_number'] == 39:
        print('\n[7, 70%][7, 40%] [all bands temporal] + DENSE | SMALL (for MTT) w/ rnn x512')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='rnn512_dense')

    ############### FORWARD ATTENTION ###############

    elif config['model_number'] == 40:
        print('\n[7, 40%] | SMALL (for MTT) w/ attention')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention')

    elif config['model_number'] == 41:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ attention')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention')

    elif config['model_number'] == 42:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ attention')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention')

    elif config['model_number'] == 43:
        print('\n[7, 40%] | SMALL (for MTT) w/ attention + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_dense')

    elif config['model_number'] == 44:
        print('\n[7, 70%][7, 40%] | SMALL (for MTT) w/ attention + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_dense')

    elif config['model_number'] == 45:
        print('\n[7, 70%][7, 40%] [all bands temporal] | SMALL (for MTT) w/ attention + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_dense')

    ############### FORWARD ATTENTION + POSITION EMBEDDING ###############

    elif config['model_number'] == 50:
        print('\n[7, 40%] | Attention + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin')

    elif config['model_number'] == 51:
        print('\n[7, 70%][7, 40%] | Attention + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin')

    elif config['model_number'] == 52:
        print('\n[7, 70%][7, 40%] [all bands temporal] | Attention + position embedding')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin')

    elif config['model_number'] == 53:
        print('\n[7, 40%] | Attention + position embedding + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin_dense')

    elif config['model_number'] == 5315:
        print('\n[7, 40%] | Attention + position embedding + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=7, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin_dense')

    elif config['model_number'] == 533:
        print('\n[7, 40%] | Attention + position embedding + dense BIG BACKEND')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=512, output_units=500, type='attention_position_sin_dense')

    elif config['model_number'] == 539:
        print('\n[7, 40%] | Attention + position embedding + dense FLEXIBLE INPUT')
        print('-----------------------------------\n')
        #out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=4.5, type='74timbral')
        print('input: ' + str(x.get_shape))
        expanded_layer = tf.expand_dims(x, 3)
        input_layer = tf.layers.batch_normalization(expanded_layer, training=is_training)
        print('input_layer: ', input_layer.get_shape())
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        conv = tf.layers.conv2d(inputs=input_pad_7,
                            filters=int(4.5*128),
                            kernel_size=[7, int(0.4 * config['yInput'])],
                            padding="valid",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn_conv = tf.layers.batch_normalization(conv, training=is_training)
        print('Output timbral block: ', bn_conv.get_shape())

        #bn_conv_pad_7 = tf.pad(bn_conv, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
        conv1 = tf.layers.conv2d(inputs=bn_conv,
                            filters=int(4.5*128),
                            kernel_size=[1, bn_conv.shape[2]],
                            padding="valid",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        print('Output "mp": ', bn_conv1.get_shape())

        out = tf.squeeze(bn_conv1, [2])
        print('Output timbral block: ', out.get_shape())
        return models_temporal_pool.backend(out, is_training, config, num_filt=64, output_units=200, type='attention_position_sin_dense')

    elif config['model_number'] == 54:
        print('\n[7, 70%][7, 40%] | Attention + position embedding + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.95, type='7774timbral')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin_dense')

    elif config['model_number'] == 55:
        print('\n[7, 70%][7, 40%] [all bands temporal] | Attention + position embedding + dense')
        print('-----------------------------------\n')
        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1.6, type='7774timbraltemporal')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin_dense')

#    elif config['model_number'] == 51: # good results!!!!!!!!!!!!!!!!!!!!!!!!
#        print('\n[7, 70%][7, 40%] + temporal | Attention + position | DENSE')
#        print('-----------------------------------\n')
#        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1, type='7774timbraltemporal')
#        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_dense')

#    elif config['model_number'] == 52: # good results!!!!!!!!!!!!!!!!!!!!!!!
#        print('\n[7, 70%][7, 40%] + temporal | Attention + position sin | DENSE')
#        print('-----------------------------------\n')
#        out_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=1, type='7774timbraltemporal')	
#        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='attention_position_sin_dense')

#    elif config['model_number'] == 68:
#        print('\n[7, 70%][7, 40%] | [7, v][7, 70%][7, 40%] + temporal |  Global pooling')
#        print('-----------------------------------\n')
#        pre_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=0.5, type='7774timbral')	
#        out_frontend = models_musically_motivated.frontend(pre_frontend, is_training, config, num_filt=0.5, type='v77774timbraltemporal')	
#        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    ############### TRANSFORMER ###############

    elif config['model_number'] == 100:
        print('\n[7, 70%][7, 40%] + temporal | Transformer | Global pooling')
        print('-----------------------------------\n')
        pre_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=0.2, type='7774timbraltemporal')
        out_frontend = models_transformer.encoder(pre_frontend, is_training, config, channels=pre_frontend.get_shape()[2], type='transformer')
        return models_temporal_pool.backend(out_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    ############### ADDITIONAL EXPERIMENTS ###############

    elif config['model_number'] == 169:
        print('\n[7, 70%][7, 40%] | CONCAT | [7, v][7, 70%][7, 40%] + temporal |  Global pooling')
        print('-----------------------------------\n')
        pre_frontend = models_musically_motivated.frontend(x, is_training, config, num_filt=0.9, type='7774timbral')
        print('pre_frontend',pre_frontend.get_shape())	
        out_frontend = models_musically_motivated.frontend(pre_frontend, is_training, config, num_filt=0.9, type='v77774timbraltemporal')
        print('out_frontend',out_frontend.get_shape())	
        concat_frontend = tf.concat([pre_frontend,out_frontend], 2)	
        return models_temporal_pool.backend(concat_frontend, is_training, config, num_filt=64, output_units=200, type='globalpool')

    raise RuntimeError("ERROR: Model {} can't be found!".format(config["model_number"]))

