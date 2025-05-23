from argparse import ArgumentParser, Namespace
from logging import DEBUG as LOG_LEVEL
from logging import basicConfig, info
from pathlib import Path
from sys import stdout

from lark import Lark

from evolvable_neuron.cloning import Cloner, ToPython
from evolvable_neuron.cloning import __file__ as cloning_module_path


def get_args() -> Namespace:

    def_pull_factor = 0.05

    argparser = ArgumentParser()
    argparser.add_argument("--src-impl-path")
    argparser.add_argument("--dst-impl-path")
    argparser.add_argument("--mutation-rate", type=float)
    argparser.add_argument("--must-mutate", action="store_true")
    argparser.add_argument("--min-cloning-iters", type=int)
    argparser.add_argument(
        "--new-expr-pull-factor",
        type=float,
        default=def_pull_factor,
        help=f"Defaults to {def_pull_factor}",
    )
    argparser.add_argument("--random-seed", type=int)

    args = argparser.parse_args()
    args.min_cloning_iters = args.min_cloning_iters or 1
    args.mutation_rate = args.mutation_rate or 0
    return args


def main():
    basicConfig(level=LOG_LEVEL)
    args = get_args()
    parser = Lark.open(Path(cloning_module_path).parent / "dense_neuron_impl.lark")

    with open(args.src_impl_path) as stream:
        info("I 'll parse %s according to the %s grammar", args.src_impl_path, parser.source_path)
        tree = parser.parse(stream.read())

    cloner = Cloner(
        grammar=parser.grammar,
        mutation_rate=args.mutation_rate,
        new_expr_pull_factor=args.new_expr_pull_factor,
        seed=args.random_seed,
    )

    info(
        "I 'll clone with a mutation rate of %f. Seed is %s and the pull factor is %f",
        cloner.mutation_rate,
        args.random_seed,
        cloner.new_expr_pull_factor,
    )

    cloning_iter = 1
    has_mutated = False

    while args.must_mutate and not has_mutated or cloning_iter <= args.min_cloning_iters:
        info("Cloning iteration %d", cloning_iter)
        # We are not removing any unused variables if the minimum number of cloning iteratios has
        # been reached, because the definition of these variables might be the only mutation that
        # has happened so far- if we undo them, then
        cloner.remove_unused_variables = cloning_iter >= args.min_cloning_iters
        tree = cloner.transform(tree)
        has_mutated = has_mutated or cloner.has_mutated
        cloning_iter += 1

    with open(args.dst_impl_path, "w") if args.dst_impl_path else stdout as stream:
        info("I 'll dump the transformed tree to %s", args.dst_impl_path or "stdout")
        ToPython(stream=stream).visit_topdown(tree)


if __name__ == "__main__":
    main()
