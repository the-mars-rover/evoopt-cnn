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


def _base():
    return random.choice(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'Ftrl'])
_base_index = 0


def _learning_rate():
    return random.choice(numpy.logspace(-5, 0, num=1000))
_learning_rate_index = 1


def _momentum():
    return random.choice([0, random.choice(numpy.linspace(0, 0.9, num=1000))])
_momentum_index = 2


def _nesterov():
    return random.choice([True, False])
_nesterov_index = 3


def _amsgrad():
    return random.choice([True, False])
_amsgrad_index = 4


def _weight_decay():
    return random.choice([None, random.choice(numpy.logspace(-4, -2, num=1000))])
_weight_decay_index = 5


def _use_ema():
    return random.choice([True, False])
_use_ema_index = 6


def _ema_momentum():
    return random.choice(numpy.linspace(0.9, 0.9999, num=1000))
_ema_momentum_index = 7


def _rho():
    return random.choice(numpy.linspace(0.85, 0.99, num=1000))
_rho_index = 8


def _epsilon():
    return random.choice(numpy.logspace(-10, -4, num=1000))
_epsilon_index = 9


def _centered():
    return random.choice([True, False])
_centered_index = 10


def _beta_1():
    return random.choice(numpy.linspace(0.8, 0.999, num=1000))
_beta_1_index = 11


def _beta_2():
    return random.choice(numpy.linspace(0.9, 0.9999, num=1000))
_beta_2_index = 12


def _learning_rate_power():
    return random.choice(numpy.linspace(-1, 0, 1000))
_learning_rate_power_index = 13


def _initial_accumulator_value():
    return random.choice(numpy.linspace(0.01, 1.0, 1000))
_initial_accumulator_value_index = 14


def _l1_regularization_strength():
    return random.choice(numpy.linspace(0.0, 0.1, 1000))
_l1_regularization_strength_index = 15


def _l2_regularization_strength():
    return random.choice(numpy.linspace(0.0, 0.1, 1000))
_l2_regularization_strength_index = 16


def _l2_shrinkage_regularization_strength():
    return random.choice(numpy.linspace(0.0, 0.1, 1000))
_l2_shrinkage_regularization_strength_index = 17


def _beta():
    return random.choice(numpy.linspace(0.0, 1.0, 1000))
_beta_index = 18


# Public utility method to print an individual.
def individual_string(individual):
    return '[Base=' + individual[_base_index] \
        + ', learning_rate=' + str(individual[_learning_rate_index]) \
        + ', momentum=' + str(individual[_momentum_index]) \
        + ' ,nesterov=' + str(individual[_nesterov_index]) \
        + ' ,amsgrad=' + str(individual[_amsgrad_index]) \
        + ' ,weight_decay=' + str(individual[_weight_decay_index]) \
        + ' ,use_ema=' + str(individual[_use_ema_index]) \
        + ' ,ema_momentum=' + str(individual[_ema_momentum_index]) \
        + ' ,rho=' + str(individual[_rho_index]) \
        + ' ,epsilon=' + str(individual[_epsilon_index]) \
        + ' ,centered=' + str(individual[_centered_index]) \
        + ' ,beta_1=' + str(individual[_beta_1_index]) \
        + ' ,beta_2=' + str(individual[_beta_2_index]) \
        + ' ,learning_rate_power=' + str(individual[_learning_rate_power_index]) \
        + ' ,initial_accumulator_value=' + str(individual[_initial_accumulator_value_index]) \
        + ' ,l1_regularization_strength=' + str(individual[_l1_regularization_strength_index]) \
        + ' ,l2_regularization_strength=' + str(individual[_l2_regularization_strength_index]) \
        + ' ,l2_shrinkage_regularization_strength=' + str(individual[_l2_shrinkage_regularization_strength_index]) \
        + ' ,beta=' + str(individual[_beta_index]) + ']'


# The method to get the NN Optimizer given an individual. We make this function public since we may later want to get
# an optimizer from a list again when analyzing results.
def get_optimizer(individual) -> keras.optimizers.Optimizer:
    logging.info('Building the Keras optimizer to use for %s.', individual_string(individual))

    if individual[_base_index] == "SGD":
        logging.info('Building SGD optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.SGD(learning_rate=individual[_learning_rate_index],
                                    momentum=individual[_momentum_index],
                                    nesterov=individual[_nesterov_index],
                                    weight_decay=individual[_weight_decay_index],
                                    use_ema=individual[_use_ema_index],
                                    ema_momentum=individual[_ema_momentum_index])

    if individual[_base_index] == "RMSprop":
        logging.info('Building RMSprop optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.RMSprop(learning_rate=individual[_learning_rate_index],
                                        rho=individual[_rho_index],
                                        momentum=individual[_momentum_index],
                                        epsilon=individual[_epsilon_index],
                                        centered=individual[_centered_index],
                                        weight_decay=individual[_weight_decay_index],
                                        use_ema=individual[_use_ema_index],
                                        ema_momentum=individual[_ema_momentum_index])

    if individual[_base_index] == "Nadam":
        logging.info('Building Nadam optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Nadam(learning_rate=individual[_learning_rate_index],
                                      beta_1=individual[_beta_1_index],
                                      beta_2=individual[_beta_2_index],
                                      epsilon=individual[_epsilon_index],
                                      weight_decay=individual[_weight_decay_index],
                                      use_ema=individual[_use_ema_index],
                                      ema_momentum=individual[_ema_momentum_index])
    if individual[_base_index] == "Ftrl":
        logging.info('Building Ftrl optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Ftrl(learning_rate=individual[_learning_rate_index],
                                     learning_rate_power=individual[_learning_rate_power_index],
                                     initial_accumulator_value=individual[_initial_accumulator_value_index],
                                     l1_regularization_strength=individual[_l1_regularization_strength_index],
                                     l2_regularization_strength=individual[_l2_regularization_strength_index],
                                     l2_shrinkage_regularization_strength=individual[_l2_shrinkage_regularization_strength_index],
                                     beta=individual[_beta_index],
                                     weight_decay=individual[_weight_decay_index],
                                     use_ema=individual[_use_ema_index],
                                     ema_momentum=individual[_ema_momentum_index])
    if individual[_base_index] == "Adamax":
        logging.info('Building Adamax optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adamax(learning_rate=individual[_learning_rate_index],
                                       beta_1=individual[_beta_1_index],
                                       beta_2=individual[_beta_2_index],
                                       epsilon=individual[_epsilon_index],
                                       weight_decay=individual[_weight_decay_index],
                                       use_ema=individual[_use_ema_index],
                                       ema_momentum=individual[_ema_momentum_index])
    if individual[_base_index] == "Adam":
        logging.info('Building Adam optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adam(learning_rate=individual[_learning_rate_index],
                                     beta_1=individual[_beta_1_index],
                                     beta_2=individual[_beta_2_index],
                                     epsilon=individual[_epsilon_index],
                                     amsgrad=[_amsgrad_index],
                                     weight_decay=individual[_weight_decay_index],
                                     use_ema=individual[_use_ema_index],
                                     ema_momentum=individual[_ema_momentum_index])
    if individual[_base_index] == "Adagrad":
        logging.info('Building Adagrad optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adagrad(learning_rate=individual[_learning_rate_index],
                                        initial_accumulator_value=individual[_initial_accumulator_value_index],
                                        epsilon=individual[_epsilon_index],
                                        weight_decay=individual[_weight_decay_index],
                                        use_ema=individual[_use_ema_index],
                                        ema_momentum=individual[_ema_momentum_index])
    if individual[_base_index] == "Adadelta":
        logging.info('Building Adadelta optimizer to use for %s.', individual_string(individual))
        return keras.optimizers.Adadelta(learning_rate=individual[_learning_rate_index],
                                         rho=individual[_rho_index],
                                         epsilon=individual[_epsilon_index],
                                         weight_decay=individual[_weight_decay_index],
                                         use_ema=individual[_use_ema_index],
                                         ema_momentum=individual[_ema_momentum_index])

    logging.error("individual[_base_index] is '" + individual[
        _base_index] + "' but must be one of \'SGD\', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', or 'Ftrl'")
    raise ValueError("individual[_base_index] is '" + individual[
        _base_index] + "' but must be one of \'SGD\', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', or 'Ftrl'")


def _register_individual():
    # The existing optimizer on which this chromosome is based. Can be one of SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, or Ftrl.
    toolbox.register("base", _base)

    # A floating point value. The learning rate. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    toolbox.register("learning_rate", _learning_rate)

    # float (>= 0). Accelerates gradient descent in the relevant direction and dampens oscillations. Applicable to SGD, RMSprop'
    toolbox.register("momentum", _momentum)

    # Boolean. Whether to apply Nesterov momentum. Applicable to SGD
    toolbox.register("nesterov", _nesterov)

    # Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond". Applicable to Adam
    toolbox.register("amsgrad", _amsgrad)

    # Float, defaults to None. If set, weight decay is applied. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    toolbox.register("weight_decay", _weight_decay)

    # Boolean. If True, exponential moving average (EMA) is applied. EMA consists of computing an exponential moving average of the weights of the model (as the weight values change after each training batch), and periodically overwriting the weights with their moving average. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    toolbox.register("use_ema", _use_ema)

    # Float. Only used if use_ema=True. This is # noqa: E501 the momentum to use when computing the EMA of the model's weights: new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value. Applicable to SGD, RMSprop, Nadam, Ftrl, Adamax, Adam, Adagrad, Adadelta'
    toolbox.register("ema_momentum", _ema_momentum)

    # float, defaults to 0.9. Discounting factor for the old gradients. Applicable to RMSprop, Adadelta'
    toolbox.register("rho", _rho)

    # A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7. Applicable to RMSprop, Nadam, Adamax, Adam, Adagrad, Adadelta'
    toolbox.register("epsilon", _epsilon)

    # Boolean. If True, gradients are normalized by the estimated variance of the gradient; if False, by the uncentered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. Defaults to False. Applicable to RMSprop'
    toolbox.register("centered", _centered)

    # A float value. The exponential decay rate for the 1st moment estimates. Defaults to 0.9. Applicable to Nadam, Adamax, Adam'
    toolbox.register("beta_1", _beta_1)

    # A float value. The exponential decay rate for the 2nd moment estimates. Defaults to 0.999. Applicable to Nadam, Adamax, Adam'
    toolbox.register("beta_2", _beta_2)

    # A float value. Must be less or equal to zero. Controls how the learning rate decreases during training. Use zero for a fixed learning rate. Applicable to Ftrl'
    toolbox.register("learning_rate_power", _learning_rate_power)

    # A float value. The starting value for accumulators. Only zero or positive values are allowed. Applicable to Ftrl, Adagrad'
    toolbox.register("initial_accumulator_value", _initial_accumulator_value)

    # A float value, must be greater than or equal to zero. Defaults to 0.0. Applicable to Ftrl'
    toolbox.register("l1_regularization_strength", _l1_regularization_strength)

    # A float value, must be greater than or equal to zero. Defaults to 0.0. Applicable to Ftrl'
    toolbox.register("l2_regularization_strength", _l2_regularization_strength)

    # A float value, must be greater than or equal to zero. This differs from L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty. When input is sparse shrinkage will only happen on the active weights. Applicable to Ftrl'
    toolbox.register("l2_shrinkage_regularization_strength", random.choice, _l2_shrinkage_regularization_strength)

    # A float value,  representing the beta value from the paper. Defaults to 0.0. Applicable to Ftrl'
    toolbox.register("beta", _beta)

    logging.info('Registering individual initialization method.')
    toolbox.register("individual", tools.initCycle, creator.Individual, (
        toolbox.base,
        toolbox.learning_rate,
        toolbox.momentum,
        toolbox.nesterov,
        toolbox.amsgrad,
        toolbox.weight_decay,
        toolbox.use_ema,
        toolbox.ema_momentum,
        toolbox.rho,
        toolbox.epsilon,
        toolbox.centered,
        toolbox.beta_1,
        toolbox.beta_2,
        toolbox.learning_rate_power,
        toolbox.initial_accumulator_value,
        toolbox.l1_regularization_strength,
        toolbox.l2_regularization_strength,
        toolbox.l2_shrinkage_regularization_strength,
        toolbox.beta,
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

    # Open a strategy scope. Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    with strategy.scope():
        optimizer = get_optimizer(individual)

        model = models.get_model(model_name, input_shape, num_classes)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

    # Return the accuracy of the model as the fitness
    score = model.evaluate(x_test, y_test, verbose=1)

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
    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_base_index] = _base()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_learning_rate_index] = _learning_rate()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_momentum_index] = _momentum()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_nesterov_index] = _nesterov()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_amsgrad_index] = _amsgrad()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_weight_decay_index] = _weight_decay()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_use_ema_index] = _use_ema()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_ema_momentum_index] = _ema_momentum()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_rho_index] = _rho()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_epsilon_index] = _epsilon()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_centered_index] = _centered()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_beta_1_index] = _beta_1()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_beta_2_index] = _beta_2()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_learning_rate_power_index] = _learning_rate_power()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_initial_accumulator_value_index] = _initial_accumulator_value()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_l1_regularization_strength_index] = _l1_regularization_strength()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_l2_shrinkage_regularization_strength_index] = _l2_shrinkage_regularization_strength()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_l2_shrinkage_regularization_strength_index] = _l2_shrinkage_regularization_strength()

    r = random.uniform(0, 1)
    if r < gene_mut_prob:
        individual[_beta_index] = _beta()

    return [individual]


def _register_genetic_operators(gene_mut_prob):
    logging.info('Registering the genetic operators.')
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate_optimizer, gene_mut_prob=gene_mut_prob)


# ======================================================================================================================
# ========================================= EVOLUTIONARY ALGORITHM =====================================================
# ======================================================================================================================

def ea_simple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """The following Evolutionary algorithm is actually taken from the DEAP library. We just have our own implementation
    to add additional logging. See https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.eaSimple."""
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    logging.info('Evaluating fitness for the initial generation of individuals.')
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    logging.info('Will evaluate fitness for %s individuals.', len(invalid_ind))
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        logging.info('Applying selection operators for generation %s.', gen)
        offspring = toolbox.select(population, len(population))

        logging.info('Applying genetic operators for generation %s.', gen)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        logging.info('Evaluating fitness for for generation %s.', gen)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logging.info('Will evaluate fitness for %s individuals.', len(invalid_ind))
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# ======================================================================================================================
# ========================================= RUN ALGORITHM ==============================================================
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
    pop, log = ea_simple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                   stats=stats, halloffame=hof, verbose=True)

    # Return the hall of fame and results
    return hof, log


__all__ = ['run', 'get_optimizer', 'individual_string']
