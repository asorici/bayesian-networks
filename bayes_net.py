from typing import Dict, Iterator, List
from collections import OrderedDict
from itertools import product
from argparse import ArgumentParser, Namespace
import numpy as np


class BayesNet:

    def __init__(self):
        self.parents = OrderedDict({})  # type: OrderedDict[str, tuple]

    def __len__(self) -> int:
        return len(self.parents)

    def add_variable(self, var_name: str, parents: List[str]) -> None:
        self.parents[var_name] = tuple(parents)

    def variables(self) -> List[str]:
        return list(self.parents.keys())

    def pcond(self, var_name: str, parent_values: tuple) -> float:
        raise NotImplementedError

    def log_prob(self, sample: Dict[str, int]) -> float:
        logprob = 0
        for var_name, parents in self.parents.items():
            parent_values = tuple(sample[p] for p in parents)
            prob = self.pcond(var_name, parent_values)
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
                    prob = self.pcond(var_name, parent_values)
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
                line += f" {self.pcond(var_name, parent_values):.2f}"

            all_lines += line + "\n"
        return all_lines


class ParametricBayesNet(BayesNet):

    def __init__(self) -> None:
        super(ParametricBayesNet, self).__init__()
        self.scores = {}  # type: Dict[str, Dict[tuple, float]]

    def add_variable(self, var_name, parents):
        super(ParametricBayesNet, self).add_variable(var_name, parents)
        var_scores = self.scores[var_name] = dict({})
        for values in product(*([[0, 1]] * len(parents))):
            var_scores[tuple(values)] = np.random.randn()

    def pcond(self, var_name: str, parent_values: tuple) -> float:
        return 1. / (1. + np.exp(-self.scores[var_name][parent_values]))

    def learn(self, sample: Dict[str, int],
              learning_rate: float = 1e-3) -> None:
        for var_name, parents in self.parents.items():
            parent_values = tuple(sample[p] for p in parents)
            prob = self.pcond(var_name, parent_values)
            grad = sample[var_name] - prob
            self.scores[var_name][parent_values] += learning_rate * grad


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

    def pcond(self, var_name: str, parent_values: tuple) -> float:
        return self.cpds[var_name][parent_values]


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


def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--file",
                            type=str,
                            default="bn1",
                            dest="file_name",
                            help="Input file")
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
    parametric_bn = read_parametric_bn(args.file_name)

    print("Initial params:")
    print(parametric_bn)

    ref_cent = cross_entropy(table_bn, table_bn)
    cent = cross_entropy(table_bn, parametric_bn, nsamples=100)
    print(f"Step {0:6d} | CE: {cent:6.3f} / {ref_cent:6.3f}")

    for step in range(1, args.nsteps + 1):
        sample = table_bn.sample()
        parametric_bn.learn(sample, learning_rate=args.lr)

        if step % 500 == 0:
            cent = cross_entropy(table_bn, parametric_bn, nsamples=200)
            print(f"Step {step:6d} | CE: {cent:6.3f} / {ref_cent:6.3f}")

    print("Final params:")
    print(parametric_bn)

    print("Reference network:")
    print(table_bn)


if __name__ == "__main__":
    main()
