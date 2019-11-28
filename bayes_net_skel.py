from typing import Dict, Iterator, List
from collections import OrderedDict
from itertools import product
from argparse import ArgumentParser, Namespace
import numpy as np
import os

class BayesNet:
    def __init__(self):
        self.parents = OrderedDict({})  # type: OrderedDict[str, tuple]

    def __len__(self) -> int:
        return len(self.parents)

    def add_variable(self, var_name: str, parents: List[str]) -> None:
        self.parents[var_name] = tuple(parents)

    def variables(self) -> List[str]:
        return list(self.parents.keys())

    def prob(self, var_name: str, parent_values: tuple) -> float:
        raise NotImplementedError

    def log_prob(self, sample: Dict[str, int]) -> float:
        logprob = 0
        for var_name, parents in self.parents.items():
            parent_values = tuple(sample[p] for p in parents)
            prob = self.prob(var_name, parent_values)
            if sample[var_name] == 0:
                prob = 1 - prob
            logprob += np.log(prob)
        return logprob

    def sample(self) -> Dict[str, int]:
        values = {}  # type: Dict[str, int]
        left_vars = self.variables()
        while left_vars:
            new_vars = []
            for var_name in left_vars:
                parents = self.parents[var_name]
                if all(p in values for p in parents):
                    parent_values = tuple(values[p] for p in parents)
                    prob = self.prob(var_name, parent_values)
                    values[var_name] = int(np.random.sample() <= prob)
                else:
                    new_vars.append(var_name)
            left_vars = new_vars
        return values

    def __str__(self) -> str:
        all_lines = ""
        for var_name, parents in self.parents.items():
            line = var_name + " ; " + " ".join(parents)
            for parent_values in product(*([[0, 1]] * len(parents))):
                #line += f" {self.prob(var_name, parent_values):.2f}"
                line += "%.2f" % self.prob(var_name, parent_values)

            all_lines += line + "\n"
        return all_lines


class TableBayesNet(BayesNet):
    def __init__(self) -> None:
        super(TableBayesNet, self).__init__()
        self.cpds = {}  # type: Dict[str, Dict[tuple, float]]

    def add_variable(self, var_name, parents):
        super(TableBayesNet, self).add_variable(var_name, parents)
        self.cpds[var_name] = dict({})

    def put(self, var_name: str,
            parent_values: tuple,
            prob: float) -> None:
        self.cpds[var_name][parent_values] = prob

    def prob(self, var_name: str, parent_values: tuple) -> float:
        return self.cpds[var_name][parent_values]


class MLEBayesNet(BayesNet):

    def __init__(self) -> None:
        super(MLEBayesNet, self).__init__()
        self.cpds = {}  # type: Dict[str, Dict[tuple, float]]

    def add_variable(self, var_name, parents):
        super(MLEBayesNet, self).add_variable(var_name, parents)
        self.cpds[var_name] = dict({})
        for values in product(*([[0, 1]] * len(parents))):
            self.cpds[var_name][tuple(values)] = 0.5

    def put(self, var_name: str,
            parent_values: tuple,
            prob: float) -> None:
        self.cpds[var_name][parent_values] = prob

    
    def prob(self, var_name: str, parent_values: tuple) -> float:
        return self.cpds[var_name][parent_values]


    def learn_cpds(self, samples: List[Dict[str, int]],
              alpha: float = 1.) -> None:
        pass


class EMBayesNet(MLEBayesNet):
    def __init__(self) -> None:
        super(EMBayesNet, self).__init__()
        self.cpds = {}  # type: Dict[str, Dict[tuple, float]]

    def learn_cpds(self, samples_with_missing: List[Dict[str, int]],
              alpha: float = 1.):
        pass


class ParametricBayesNet(BayesNet):

    def __init__(self) -> None:
        super(ParametricBayesNet, self).__init__()
        self.scores = {}  # type: Dict[str, Dict[tuple, float]]

    def add_variable(self, var_name, parents):
        super(ParametricBayesNet, self).add_variable(var_name, parents)
        var_scores = self.scores[var_name] = dict({})
        for values in product(*([[0, 1]] * len(parents))):
            var_scores[tuple(values)] = np.random.randn()

    def prob(self, var_name: str, parent_values: tuple) -> float:
        return 1. / (1. + np.exp(-self.scores[var_name][parent_values]))

    def learn(self, sample: Dict[str, int],
              learning_rate: float = 1e-3) -> None:
        
        for var_name, parents in self.parents.items():
            key = tuple(sample[p] for p in parents)
            score = self.scores[var_name][key]
            ## TODO 2: YOUR CODE HERE

            # compute the gradient with respect to the parameters modeling the CPD of variable var_name
            # grad = ...
            
            # update the parameters
            # self.scores[var_name][key] = ...


def all_dicts(variables: List[str]) -> Iterator[Dict[str, int]]:
    for keys in product(*([[0, 1]] * len(variables))):
        yield dict(zip(variables, keys))


def cross_entropy(bn1: BayesNet, bn2: BayesNet, nsamples: int = None) -> float:
    cross_ent = .0
    if nsamples is None:
        for sample in all_dicts(bn1.variables()):
            cross_ent -= np.exp(bn1.log_prob(sample)) * bn2.log_prob(sample)
    else:
        for _ in range(nsamples):
            cross_ent -= bn2.log_prob(bn1.sample())
        cross_ent /= nsamples
    return cross_ent


def read_cpd_bn(file_name: str) -> TableBayesNet:
    bnet = TableBayesNet()
    with open(file_name, "r") as handler:
        nvars, *_ = [int(val) for val in handler.readline().split()]
        for _ in range(nvars):
            line = handler.readline()
            parts = line.split(";")
            if len(parts) != 3:
                raise RuntimeError("Strange variable line: " + line)
            var_name = parts[0].strip()
            parents = parts[1].split()
            probs = [float(val) for val in parts[2].split()]
            bnet.add_variable(var_name, parents)
            all_parent_values = product(*([[0, 1]] * len(parents)))
            for (parent_values, prob) in zip(all_parent_values, probs):
                bnet.put(var_name, parent_values, prob)
    return bnet


def read_mle_bn(file_name: str) -> MLEBayesNet:
    bnet = MLEBayesNet()
    with open(file_name, "r") as handler:
        nvars, *_ = [int(val) for val in handler.readline().split()]
        for _ in range(nvars):
            line = handler.readline()
            parts = line.split(";")
            if len(parts) != 3:
                raise RuntimeError("Strange variable line: " + line)
            var_name = parts[0].strip()
            parents = parts[1].split()
            probs = [float(val) for val in parts[2].split()]
            bnet.add_variable(var_name, parents)
    return bnet

def read_parametric_bn(file_name: str) -> ParametricBayesNet:
    bnet = ParametricBayesNet()
    with open(file_name, "r") as handler:
        nvars, *_ = [int(val) for val in handler.readline().split()]
        for _ in range(nvars):
            line = handler.readline()
            parts = line.split(";")
            if len(parts) != 3:
                raise RuntimeError("Strange variable line: " + line)
            var_name = parts[0].strip()
            parents = parts[1].split()
            bnet.add_variable(var_name, parents)
    return bnet


def read_em_bn(file_name: str) -> EMBayesNet:
    bnet = EMBayesNet()
    with open(file_name, "r") as handler:
        nvars, *_ = [int(val) for val in handler.readline().split()]
        for _ in range(nvars):
            line = handler.readline()
            parts = line.split(";")
            if len(parts) != 3:
                raise RuntimeError("Strange variable line: " + line)
            var_name = parts[0].strip()
            parents = parts[1].split()
            probs = [float(val) for val in parts[2].split()]
            bnet.add_variable(var_name, parents)
    return bnet


def read_samples(file_name: str) -> List[Dict[str, int]]:
    samples = [] 
    with open(file_name, "r") as handler:
        lines = handler.readlines()
        # read first line of file to get variables in order
        variables = [str(v) for v in lines[0].split()]

        for i in range(1, len(lines)):
            vals = [int(v) for v in lines[i].split()]
            sample = dict(zip(variables, vals))
            samples.append(sample)

    return samples



def create_samples_with_misses(bn: TableBayesNet, prob_missing: float = 0.15,
                               output_filename: str = "bn1_samples_missing", nr_samples=10000) -> None:
    bn_vars = bn.variables()

    with open(output_filename, "w") as out:
        out.write(" ".join(bn_vars) + os.linesep)

        for i in range(nr_samples):
            sample = bn.sample()

            for idx, v in enumerate(bn_vars):
                if np.random.rand() < prob_missing:
                    sample[v] = 2

                if idx != len(bn_vars) - 1:
                    out.write(str(sample[v]) + " ")
                else:
                    out.write(str(sample[v]) + os.linesep)



def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--file",
                            type=str,
                            default="bn1",
                            dest="file_name",
                            help="Input file")
    arg_parser.add_argument("-s", "--samplefile",
                            type=str,
                            default="samples_bn1",
                            dest="samples_file_name",
                            help="Samples file")
    arg_parser.add_argument("-sm", "--samplemissingfile",
                            type=str,
                            default="bn1_samples_missing",
                            dest="samples_missing_file_name",
                            help="Samples with missing values file")
    arg_parser.add_argument("-n", "--nsteps",
                            type=int,
                            default=10000,
                            dest="nsteps",
                            help="Number of optimization steps")
    arg_parser.add_argument("--lr",
                            type=float,
                            default=.005,
                            dest="lr",
                            help="Learning rate")

    return arg_parser.parse_args()


def main():
    args = get_args()

    
    table_bn = read_cpd_bn(args.file_name)
    mle_bn = read_mle_bn(args.file_name)
    parametric_bn = read_parametric_bn(args.file_name)
    em_bn = read_em_bn(args.file_name)

    print("Initial params MLE bn:")
    print(mle_bn)

    print("Initial params parametric bn:")
    print(parametric_bn)

    print("Initial params EM bn:")
    print(em_bn)

    samples = read_samples(args.samples_file_name)
    mle_bn.learn_cpds(samples)

    samples_with_missing = read_samples(args.samples_missing_file_name)
    em_bn.learn_cpds(samples_with_missing)

    ref_cent = cross_entropy(table_bn, table_bn)
    cent = cross_entropy(table_bn, parametric_bn, nsamples=100)
    #print(f"Step {0:6d} | CE: {cent:6.3f} / {ref_cent:6.3f}")
    print("Step %6d | CE: %6.3f / %6.3f" % (0, cent, ref_cent))

    for step in range(1, args.nsteps + 1):
        sample = table_bn.sample()
        parametric_bn.learn(sample, learning_rate=args.lr)

        if step % 500 == 0:
            cent = cross_entropy(table_bn, parametric_bn, nsamples=200)
            #print(f"Step {step:6d} | CE: {cent:6.3f} / {ref_cent:6.3f}")
            print("Step %6d | CE: %6.3f / %6.3f" % (step, cent, ref_cent))

    print("Reference network:")
    print(table_bn)

    print("Final params MLE Network:")
    print(mle_bn)

    print("Final params Gradient Descent Network:")
    print(parametric_bn)

    print("Final params EM Network:")
    print(em_bn)


if __name__ == "__main__":
    main()
