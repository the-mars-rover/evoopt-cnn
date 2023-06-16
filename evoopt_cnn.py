# %%file evoopt_cnn.py

import random
import numpy
import models
import tensorflow
import os
import logging

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from tensorflow import keras

# ======================================================================================================================
# ========================================= INITIAL SETUP ==============================================================
# ======================================================================================================================

# Make sure we enable memory growth for all GPUs, because we do not want to allocate all memory on the devices.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(e)

# Create a MirroredStrategy for tensorflow
strategy = tensorflow.distribute.MirroredStrategy(cross_device_ops=tensorflow.distribute.ReductionToOneDevice())

# The toolbox must be initialized here, otherwise the DEAP library does not work.
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# ======================================================================================================================
# ========================================= CHROMOSOME REPRESENTATION ==================================================
# ======================================================================================================================

_possible_base_values = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'Ftrl']
_possible_learning_rate_values = [0.1, 0.01, 0.001, 0.0001]
_possible_momentum_values = [0, 0.5, 0.9, 0.99]
_possible_nesterov_values = [True, False]
_possible_amsgrad_values = [True, False]
_possible_weight_decay_values = [None, 0.0001, 0.001, 0.01]
_possible_clipnorm_values = [None, 1, 5, 10]
_possible_clipvalue_values = [None, 0.1, 0.5, 1]
_possible_global_clipnorm_values = [None, 1, 5, 10]
_possible_use_ema_values = [True, False]
_possible_ema_momentum_values = [0.9, 0.99, 0.999]
_possible_ema_overwrite_frequency_values = [1, 5, 10, 20]
_possible_rho_values = [0.9, 0.95, 0.99]
_possible_epsilon_values = [1e-8, 1e-7, 1e-6]
_possible_centered_values = [True, False]
_possible_beta_1_values = [0.9, 0.95, 0.99]
_possible_beta_2_values = [0.999, 0.9999, 0.99999]
_possible_learning_rate_power_values = [0.0, -0.5, -1.0]
_possible_initial_accumulator_value_values = [0.0, 0.1, 1.0, 10.0]
_possible_l1_regularization_strength_values = [0.0, 0.0001, 0.001, 0.01]
_possible_l2_regularization_strength_values = [0.0, 0.1, 0.01, 0.001, 0.0001]
_possible_l2_shrinkage_regularization_strength_values = [0.0, 0.0001, 0.001, 0.01]
_possible_beta_values = [0.0, 0.1, 1.0, 10.0]


# Public utility method to print an individual.
def individual_string(individual):
    return '[Base=' + individual[0] \
        + ', learning_rate=' + str(individual[1]) \
        + ', momentum=' + str(individual[2]) \
        + ' ,nesterov=' + str(individual[3]) \
        + ' ,amsgrad=' + str(individual[4]) \
        + ' ,weight_decay=' + str(individual[5]) \
        + ' ,clipnorm=' + str(individual[6]) \
        + ' ,clipvalue=' + str(individual[7]) \
        + ' ,global_clipnorm=' + str(individual[8]) \
        + ' ,use_ema=' + str(individual[9]) \
        + ' ,ema_momentum=' + str(individual[10]) \
        + ' ,ema_overwrite_frequency=' + str(individual[11]) \
        + ' ,rho=' + str(individual[12]) \
        + ' ,epsilon=' + str(individual[13]) \
        + ' ,centered=' + str(individual[14]) \
        + ' ,beta_1=' + str(individual[15]) \
        + ' ,beta_2=' + str(individual[16]) \
        + ' ,learning_rate_power=' + str(individual[17]) \
        + ' ,initial_accumulator_value=' + str(individual[18]) \
        + ' ,l1_regularization_strength=' + str(individual[19]) \
        + ' ,l2_regularization_strength=' + str(individual[20]) \
        + ' ,l2_shrinkage_regularization_strength=' + str(individual[21]) \
        + ' ,beta=' + str(individual[22]) + ']'


# The method to get the NN Optimizer given an individual. We make this function public since we may later want to get
# an optimizer from a list again when analyzing results.
def get_optimizer(individual) -> keras.optimizers.Optimizer:
    logging.info('Building the Keras optimizer to use for %s.', individual_string(individual))

    # Only one of clipnorm, clipvalue, or global_clipnorm can have a value.
    if individual[6] is not None:
        logging.info('Since `clipnorm` is present, `clipvalue` and `global_clipnorm` are being set to `None`.')
        individual[7] = None
        individual[8] = None
    if individual[7] is not None:
        logging.info('Since `clipvalue` is present, `global_clipnorm` is being set to `None`.')
        individual[8] = None

    if individual[0] == "SGD":
        logging.info('Building SGD optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.SGD(learning_rate=individual[1],
                                    momentum=individual[2],
                                    nesterov=individual[3],
                                    weight_decay=individual[5],
                                    clipnorm=individual[6],
                                    clipvalue=individual[7],
                                    global_clipnorm=individual[8],
                                    use_ema=individual[9],
                                    ema_momentum=individual[10],
                                    ema_overwrite_frequency=individual[11])

    if individual[0] == "RMSprop":
        logging.info('Building RMSprop optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.RMSprop(learning_rate=individual[1],
                                        rho=individual[12],
                                        momentum=individual[2],
                                        epsilon=individual[13],
                                        centered=individual[14],
                                        weight_decay=individual[5],
                                        clipnorm=individual[6],
                                        clipvalue=individual[7],
                                        global_clipnorm=individual[8],
                                        use_ema=individual[9],
                                        ema_momentum=individual[10],
                                        ema_overwrite_frequency=individual[11])

    if individual[0] == "Nadam":
        logging.info('Building Nadam optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Nadam(learning_rate=individual[1],
                                      beta_1=individual[15],
                                      beta_2=individual[16],
                                      epsilon=individual[13],
                                      weight_decay=individual[5],
                                      clipnorm=individual[6],
                                      clipvalue=individual[7],
                                      global_clipnorm=individual[8],
                                      use_ema=individual[9],
                                      ema_momentum=individual[10],
                                      ema_overwrite_frequency=individual[11])
    if individual[0] == "Ftrl":
        logging.info('Building Ftrl optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Ftrl(learning_rate=individual[1],
                                     learning_rate_power=individual[17],
                                     initial_accumulator_value=individual[18],
                                     l1_regularization_strength=individual[19],
                                     l2_regularization_strength=individual[20],
                                     l2_shrinkage_regularization_strength=individual[21],
                                     beta=individual[22],
                                     weight_decay=individual[5],
                                     clipnorm=individual[6],
                                     clipvalue=individual[7],
                                     global_clipnorm=individual[8],
                                     use_ema=individual[9],
                                     ema_momentum=individual[10],
                                     ema_overwrite_frequency=individual[11])
    if individual[0] == "Adamax":
        logging.info('Building Adamax optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adamax(learning_rate=individual[1],
                                       beta_1=individual[15],
                                       beta_2=individual[16],
                                       epsilon=individual[13],
                                       weight_decay=individual[5],
                                       clipnorm=individual[6],
                                       clipvalue=individual[7],
                                       global_clipnorm=individual[8],
                                       use_ema=individual[9],
                                       ema_momentum=individual[10],
                                       ema_overwrite_frequency=individual[11])
    if individual[0] == "Adam":
        logging.info('Building Adam optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adam(learning_rate=individual[1],
                                     beta_1=individual[15],
                                     beta_2=individual[16],
                                     epsilon=individual[13],
                                     amsgrad=[4],
                                     weight_decay=individual[5],
                                     clipnorm=individual[6],
                                     clipvalue=individual[7],
                                     global_clipnorm=individual[8],
                                     use_ema=individual[9],
                                     ema_momentum=individual[10],
                                     ema_overwrite_frequency=individual[11])
    if individual[0] == "Adagrad":
        logging.info('Building Adagrad optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adagrad(learning_rate=individual[1],
                                        initial_accumulator_value=individual[18],
                                        epsilon=individual[13],
                                        weight_decay=individual[5],
                                        clipnorm=individual[6],
                                        clipvalue=individual[7],
                                        global_clipnorm=individual[8],
                                        use_ema=individual[9],
                                        ema_momentum=individual[10],
                                        ema_overwrite_frequency=individual[11])
    if individual[0] == "Adadelta":
        logging.info('Building Adadelta optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adadelta(learning_rate=individual[1],
                                         rho=individual[12],
                                         epsilon=individual[13],
                                         weight_decay=individual[5],
                                         clipnorm=individual[6],
                                         clipvalue=individual[7],
                                         global_clipnorm=individual[8],
                                         use_ema=individual[9],
                                         ema_momentum=individual[10],
                                         ema_overwrite_frequency=individual[11])

    logging.error("individual[0] is '" + individual[
        0] + "' but must be one of \'SGD\', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', or 'Ftrl'")
    raise ValueError("individual[0] is '" + individual[
        0] + "' but must be one of \'SGD\', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', or 'Ftrl'")


def _register_individual():
    # 0. The existing optimizer on which this chromosome is based. Can be one of SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, or Ftrl.
    logging.info(
        'Registering initialisation method for `base` using a random choice from the following values %s',
        _possible_base_values)
    toolbox.register("base", random.choice, _possible_base_values)

    # 1. A floating point value. The learning rate. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `learning_rate` using a random choice from the following values %s',
        _possible_learning_rate_values)
    toolbox.register("learning_rate", random.choice, _possible_learning_rate_values)

    # 2. float (>= 0). Accelerates gradient descent in the relevant direction and dampens oscillations. Applicable to SGD, RMSprop'
    logging.info(
        'Registering initialisation method for `momentum` using a random choice from the following values %s',
        _possible_momentum_values)
    toolbox.register("momentum", random.choice, _possible_momentum_values)

    # 3. Boolean. Whether to apply Nesterov momentum. Applicable to SGD
    logging.info(
        'Registering initialisation method for `nesterov` using a random choice from the following values %s',
        _possible_nesterov_values)
    toolbox.register("nesterov", random.choice, _possible_nesterov_values)

    # 4. Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond". Applicable to Adam
    logging.info(
        'Registering initialisation method for `amsgrad` using a random choice from the following values %s',
        _possible_amsgrad_values)
    toolbox.register("amsgrad", random.choice, _possible_amsgrad_values)

    # 5. Float, defaults to None. If set, weight decay is applied. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `weight_decay` using a random choice from the following values %s',
        _possible_weight_decay_values)
    toolbox.register("weight_decay", random.choice, _possible_weight_decay_values)

    # 6. Float, or None.  If set, the gradient of each weight is individually clipped so that its norm is no higher than this value. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `clipnorm` using a random choice from the following values %s',
        _possible_clipnorm_values)
    toolbox.register("clipnorm", random.choice, _possible_clipnorm_values)

    # 7. Float, or None. If set, the gradient of each weight is clipped to be no higher than this value. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `clipvalue` using a random choice from the following values %s',
        _possible_clipvalue_values)
    toolbox.register("clipvalue", random.choice, _possible_clipvalue_values)

    # 8. Float, or None. If set, the gradient of all weights is clipped so that their global norm is no higher than this value. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta
    logging.info(
        'Registering initialisation method for `global_clipnorm` using a random choice from the following values %s',
        _possible_global_clipnorm_values)
    toolbox.register("global_clipnorm", random.choice, _possible_global_clipnorm_values)

    # 9. Boolean. If True, exponential moving average (EMA) is applied. EMA consists of computing an exponential moving average of the weights of the model (as the weight values change after each training batch), and periodically overwriting the weights with their moving average. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `use_ema` using a random choice from the following values %s',
        _possible_use_ema_values)
    toolbox.register("use_ema", random.choice, _possible_use_ema_values)

    # 10. Float. Only used if use_ema=True. This is # noqa: E501 the momentum to use when computing the EMA of the model's weights: new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `ema_momentum` using a random choice from the following values %s',
        _possible_ema_momentum_values)
    toolbox.register("ema_momentum", random.choice, _possible_ema_momentum_values)

    # 11. Int or None. Only used if use_ema=True. Every ema_overwrite_frequency steps of iterations, we overwrite the model variable by its moving average. If None, the optimizer # noqa: E501 does not overwrite model variables in the middle of training, and you need to explicitly overwrite the variables at the end of training by calling optimizer.finalize_variable_values() (which updates the model # noqa: E501 variables in-place). When using the built-in fit() training loop, this happens automatically after the last epoch, and you don't need to do anything. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta
    logging.info(
        'Registering initialisation method for `ema_overwrite_frequency` using a random choice from the following values %s',
        _possible_ema_overwrite_frequency_values)
    toolbox.register("ema_overwrite_frequency", random.choice, _possible_ema_overwrite_frequency_values)

    # 12.float, defaults to 0.9. Discounting factor for the old gradients. Applicable to RMSprop, Adadelta'
    logging.info(
        'Registering initialisation method for `rho` using a random choice from the following values %s',
        _possible_rho_values)
    toolbox.register("rho", random.choice, _possible_rho_values)

    # 13. A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7. Applicable to RMSprop, Nadam, Adamax, Adam, Adagrad, Adadelta'
    logging.info(
        'Registering initialisation method for `epsilon` using a random choice from the following values %s',
        _possible_epsilon_values)
    toolbox.register("epsilon", random.choice, _possible_epsilon_values)

    # 14. Boolean. If True, gradients are normalized by the estimated variance of the gradient; if False, by the uncentered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. Defaults to False. Applicable to RMSprop'
    logging.info(
        'Registering initialisation method for `centered` using a random choice from the following values %s',
        _possible_centered_values)
    toolbox.register("centered", random.choice, _possible_centered_values)

    # 15. A float value. The exponential decay rate for the 1st moment estimates. Defaults to 0.9. Applicable to Nadam, Adamax, Adam'
    logging.info(
        'Registering initialisation method for `beta_1` using a random choice from the following values %s',
        _possible_beta_1_values)
    toolbox.register("beta_1", random.choice, _possible_beta_1_values)

    # 16. A float value. The exponential decay rate for the 2nd moment estimates. Defaults to 0.999. Applicable to Nadam, Adamax, Adam'
    logging.info(
        'Registering initialisation method for `beta_2` using a random choice from the following values %s',
        _possible_beta_2_values)
    toolbox.register("beta_2", random.choice, _possible_beta_2_values)

    # 17. A float value. Must be less or equal to zero. Controls how the learning rate decreases during training. Use zero for a fixed learning rate. Applicable to Ftrl'
    logging.info(
        'Registering initialisation method for `learning_rate_power` using a random choice from the following values %s',
        _possible_learning_rate_power_values)
    toolbox.register("learning_rate_power", random.choice, _possible_learning_rate_power_values)

    # 18. A float value. The starting value for accumulators. Only zero or positive values are allowed. Applicable to Ftrl, Adagrad'
    logging.info(
        'Registering initialisation method for `initial_accumulator_value` using a random choice from the following values %s',
        _possible_initial_accumulator_value_values)
    toolbox.register("initial_accumulator_value", random.choice, _possible_initial_accumulator_value_values)

    # 19. A float value, must be greater than or equal to zero. Defaults to 0.0. Applicable to Ftrl'
    logging.info(
        'Registering initialisation method for `l1_regularization_strength` using a random choice from the following values %s',
        _possible_l1_regularization_strength_values)
    toolbox.register("l1_regularization_strength", random.choice, _possible_l1_regularization_strength_values)

    # 20. A float value, must be greater than or equal to zero. Defaults to 0.0. Applicable to Ftrl'
    logging.info(
        'Registering initialisation method for `l2_regularization_strength` using a random choice from the following values %s',
        _possible_l2_regularization_strength_values)
    toolbox.register("l2_regularization_strength", random.choice, _possible_l2_regularization_strength_values)

    # 21. A float value, must be greater than or equal to zero. This differs from L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty. When input is sparse shrinkage will only happen on the active weights. Applicable to Ftrl'
    logging.info(
        'Registering initialisation method for `l2_shrinkage_regularization_strength` using a random choice from the following values %s',
        _possible_l2_shrinkage_regularization_strength_values)
    toolbox.register("l2_shrinkage_regularization_strength", random.choice,
                     _possible_l2_shrinkage_regularization_strength_values)

    # 22. A float value,  representing the beta value from the paper. Defaults to 0.0. Applicable to Ftrl'
    logging.info(
        'Registering initialisation method for `beta` using a random choice from the following values %s',
        _possible_beta_values)
    toolbox.register("beta", random.choice, _possible_beta_values)

    logging.info('Registering individual initialization method.')
    toolbox.register("individual", tools.initCycle, creator.Individual, (
        toolbox.base,  # 0
        toolbox.learning_rate,  # 1
        toolbox.momentum,  # 2
        toolbox.nesterov,  # 3
        toolbox.amsgrad,  # 4
        toolbox.weight_decay,  # 5
        toolbox.clipnorm,  # 6
        toolbox.clipvalue,  # 7
        toolbox.global_clipnorm,  # 8
        toolbox.use_ema,  # 9
        toolbox.ema_momentum,  # 10
        toolbox.ema_overwrite_frequency,  # 11
        toolbox.rho,  # 12
        toolbox.epsilon,  # 13
        toolbox.centered,  # 14
        toolbox.beta_1,  # 15
        toolbox.beta_2,  # 16
        toolbox.learning_rate_power,  # 17
        toolbox.initial_accumulator_value,  # 18
        toolbox.l1_regularization_strength,  # 19
        toolbox.l2_regularization_strength,  # 20
        toolbox.l2_shrinkage_regularization_strength,  # 21
        toolbox.beta  # 22
    ), n=1)


# ======================================================================================================================
# ========================================= POPULATION INITIALISATION ==================================================
# ======================================================================================================================

def _register_population_initialization():
    logging.info('Registering population initialization method.')
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ======================================================================================================================
# ========================================= FITNESS EVALUATION =========================================================
# ======================================================================================================================

def _evaluate(individual, model_name, input_shape, num_classes, x_train, y_train, x_test, y_test, batch_size, epochs):
    logging.info("Evaluating the fitness of the following individual: %s.", individual_string(individual))

    # Open a strategy scope. Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    with strategy.scope():
        logging.info("Getting the optimizer for the following individual %s.", individual_string(individual))
        optimizer = get_optimizer(individual)

        logging.info("Getting the keras model for the following individual %s.", individual_string(individual))
        model = models.get_model(model_name, input_shape, num_classes)

        logging.info("Compiling the keras model for the following individual %s.", individual_string(individual))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Train the model
    logging.info("Training the keras model for the following individual %s.", individual_string(individual))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

    # Return the accuracy of the model as the fitness
    logging.info("Evaluating the keras model for the following individual %s.", individual_string(individual))
    score = model.evaluate(x_test, y_test, verbose=1)

    logging.info("Evaluation accuracy of %s for the following individual %s.", score[1], individual_string(individual))
    return [score[1]]


def _register_evaluate(model_name, input_shape, num_classes, x_train, y_train, x_test, y_test, batch_size, epochs):
    logging.info('Registering the evaluation method.')
    toolbox.register("evaluate", _evaluate, model_name=model_name, input_shape=input_shape, num_classes=num_classes,
                     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, batch_size=batch_size,
                     epochs=epochs)


# ======================================================================================================================
# ========================================= SELECTION METHOD ===========================================================
# ======================================================================================================================

def _register_selection_method(tournsize):
    logging.info('Registering the selection method.')
    toolbox.register("select", tools.selTournament, tournsize=tournsize)


# ======================================================================================================================
# ========================================= GENETIC OPERATORS ==========================================================
# ======================================================================================================================

def _mutate_optimizer(individual, gene_mut_prob):
    logging.info('Mutating the following individual: %s.', individual_string(individual))

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `base` gene for following individual: %s.', individual_string(individual))
        individual[0] = random.choice(_possible_base_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `learning_rate` gene for following individual: %s.',
                     individual_string(individual))
        individual[1] = random.choice(_possible_learning_rate_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `momentum` gene for following individual: %s.',
                     individual_string(individual))
        individual[2] = random.choice(_possible_momentum_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `nesterov` gene for following individual: %s.',
                     individual_string(individual))
        individual[3] = random.choice(_possible_nesterov_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `amsgrad` gene for following individual: %s.',
                     individual_string(individual))
        individual[4] = random.choice(_possible_amsgrad_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `weight_decay` gene for following individual: %s.',
                     individual_string(individual))
        individual[5] = random.choice(_possible_weight_decay_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `clipnorm` gene for following individual: %s.',
                     individual_string(individual))
        individual[6] = random.choice(_possible_clipnorm_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `clipvalue` gene for following individual: %s.',
                     individual_string(individual))
        individual[7] = random.choice(_possible_clipvalue_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `global_clipnorm` gene for following individual: %s.',
                     individual_string(individual))
        individual[8] = random.choice(_possible_global_clipnorm_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `use_ema` gene for following individual: %s.',
                     individual_string(individual))
        individual[9] = random.choice(_possible_use_ema_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `ema_momentum` gene for following individual: %s.',
                     individual_string(individual))
        individual[10] = random.choice(_possible_ema_momentum_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `ema_overwrite_frequency` gene for following individual: %s.',
                     individual_string(individual))
        individual[11] = random.choice(_possible_ema_overwrite_frequency_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `rho` gene for following individual: %s.',
                     individual_string(individual))
        individual[12] = random.choice(_possible_rho_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `epsilon` gene for following individual: %s.',
                     individual_string(individual))
        individual[13] = random.choice(_possible_epsilon_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `centered` gene for following individual: %s.',
                     individual_string(individual))
        individual[14] = random.choice(_possible_centered_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `beta_1` gene for following individual: %s.',
                     individual_string(individual))
        individual[15] = random.choice(_possible_beta_1_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `beta_2` gene for following individual: %s.',
                     individual_string(individual))
        individual[16] = random.choice(_possible_beta_2_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `learning_rate_power` gene for following individual: %s.',
                     individual_string(individual))
        individual[17] = random.choice(_possible_learning_rate_power_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `initial_accumulator_value` gene for following individual: %s.',
                     individual_string(individual))
        individual[18] = random.choice(_possible_initial_accumulator_value_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `l1_regularization_strength` gene for following individual: %s.',
                     individual_string(individual))
        individual[19] = random.choice(_possible_l1_regularization_strength_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `l2_regularization_strength` gene for following individual: %s.',
                     individual_string(individual))
        individual[20] = random.choice(_possible_l2_regularization_strength_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `l2_shrinkage_regularization_strength` gene for following individual: %s.',
                     individual_string(individual))
        individual[21] = random.choice(_possible_l2_shrinkage_regularization_strength_values)

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        logging.info('Mutating the `beta` gene for following individual: %s.',
                     individual_string(individual))
        individual[22] = random.choice(_possible_beta_values)

    return [individual]


def _register_genetic_operators(gene_mut_prob):
    logging.info('Registering the genetic operators.')
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate_optimizer, gene_mut_prob=gene_mut_prob)


# ======================================================================================================================
# ========================================= EVOLUTIONARY ALGORITHM =====================================================
# ======================================================================================================================

def run(model_name, input_shape, num_classes, x_train, y_train, x_test, y_test, tournsize, batch_size, epochs,
        gene_mut_prob, pop_size, cxpb, mutpb, ngen, multiprocessing_pool):
    logging.info('Setting up DEAP toolbox.')
    _register_individual()
    _register_population_initialization()
    _register_selection_method(tournsize)
    _register_evaluate(model_name, input_shape, num_classes, x_train, y_train, x_test, y_test, batch_size, epochs)
    _register_genetic_operators(gene_mut_prob)
    toolbox.register("map", multiprocessing_pool.map)

    logging.info('Setting up the hall of fame and stats we want to keep track of.')
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logging.info('Initializing the initial population.')
    pop = toolbox.population(n=pop_size)

    logging.info('Running the evolutionary algorithm.')
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                   stats=stats, halloffame=hof, verbose=True)

    # Return the hall of fame and results
    return hof, log


__all__ = ['run', 'get_optimizer', 'individual_string']
