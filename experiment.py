# %%file experiment.py

import pickle
import random
import argparse
import evoopt_cnn
import datasets

# Setup the argument parser for this script
parser = argparse.ArgumentParser(description='Run an experiment using the EvoOpt-CNN algorithm.')
parser.add_argument('--results_path', dest='results_path', type=str, default='./experiment_results',
                    help='path to the folder where result files will be stored (defaults to \'./experiment_results\')')
parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='a seed that can be used in future to produce the same results (defaults to 1)')
parser.add_argument('--dataset', dest='dataset', choices=['mnist', 'fashion_mnist'], default='mnist',
                    help='the dataset to run the experiment on (defaults to \'mnist\')')
parser.add_argument('--pop_size', dest='pop_size', type=int, default=100,
                    help='the size of the population to evolve (defaults to 100)')
parser.add_argument('--ngen', dest='ngen', type=int, default=10,
                    help='the number of generations to evolve the population (defaults to 10)')
parser.add_argument('--model', dest='model', choices=['alexnet'], default='alexnet',
                    help='the NN architectural model to use to train neural networks with individual optimizers when evaluating fitness (defaults to \'alexnet\')')
parser.add_argument('--epochs', dest='epochs', type=int, default=10,
                    help='the number of epochs to train neural networks with individual optimizers when evaluating fitness (defaults to 10)')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='the mini batch size to use to train neural networks with individual optimizers when evaluating fitness (defaults to 128)')
parser.add_argument('--tournsize', dest='tournsize', type=int, default=10,
                    help='the size of the tournament for tournament selection (defaults to 10)')
parser.add_argument('--cxpb', dest='cxpb', type=float, default=0.5,
                    help='the crossover probability (defaults to 0.5)')
parser.add_argument('--mutpb', dest='mutpb', type=float, default=0.2,
                    help='the mutation probability (defaults to 0.2)')
parser.add_argument('--gene_mut_prob', dest='gene_mut_prob', type=float, default=0.5,
                    help='the probability that a gene will be mutated when mutation takes place (defaults to 0.5)')

if __name__ == '__main__':
    print('EvoOpt Experiment >>> starting experiment.')

    print('EvoOpt Experiment >>> setting experiment arguments.')
    args = parser.parse_args()
    print('EvoOpt Experiment >>> experiment arguments set and printed below.')
    print(args)

    print('EvoOpt Experiment >>> setting the random number generator seed for this experiment.')
    random.seed(args.seed)
    print('EvoOpt Experiment >>> random seed has been set.')

    print('EvoOpt Experiment >>> loading dataset for the experiment.')
    (input_shape, num_classes), (x_train, y_train), (x_test, y_test) = datasets.load_dataset(dataset_name=args.dataset)
    print('EvoOpt Experiment >>> dataset has been loaded.')

    print("EvoOpt Experiment >>> running the evolutionary algorithm with the given hyper-parameters. This may take a while. Statistics for every generation will be printed below.")
    hof, log = evoopt_cnn.run(
        model_name=args.model, input_shape=input_shape, num_classes=num_classes,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, tournsize=args.tournsize,
        batch_size=args.batch_size, epochs=args.epochs, gene_mut_prob=args.gene_mut_prob,
        pop_size=args.pop_size, cxpb=args.cxpb, mutpb=args.mutpb, ngen=args.ngen)
    print('EvoOpt Experiment >>> evolutionary algorithm has completed successfully.')

    print('EvoOpt Experiment >>> saving the results to the folder specified in the arguments.')
    file = open(args.results_path + '/' + str(args.seed) + '.arg', 'wb')
    pickle.dump(args, file)
    file.close()
    file = open(args.results_path + '/' + str(args.seed) + '.log', 'wb')
    pickle.dump(log, file)
    file.close()
    file = open(args.results_path + '/' + str(args.seed) + '.hof', 'wb')
    pickle.dump(hof, file)
    file.close()
    print('EvoOpt Experiment >>> experiment results have been saved to the specified folder.')

    print('EvoOpt Experiment >>> experiment finished.')
