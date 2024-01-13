from argparse import ArgumentParser, Namespace
from logging import DEBUG, basicConfig, info
from pathlib import Path

from lark import Lark

from evolvable_neuron.cloning import Cloner, ToPython
from evolvable_neuron.cloning import __file__ as cloning_module_path


def get_args() -> Namespace:
    argparser = ArgumentParser()
    argparser.add_argument("--src-impl-path")
    argparser.add_argument("--dst-impl-path")
    argparser.add_argument("--mutation-rate", type=float)
    argparser.add_argument("--must-mutate", action="store_true")
    argparser.add_argument("--min-cloning-iters", type=int)
    argparser.add_argument("--new-expr-pull-factor", type=float)
    argparser.add_argument("--random-seed", type=int)

    args = argparser.parse_args()
    args.min_cloning_iters = args.min_cloning_iters or 1
    args.mutation_rate = args.mutation_rate or 0
    return args


def main():
    basicConfig(level=DEBUG)
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
        cloner.reset()
        tree = cloner.transform(tree)
        has_mutated = has_mutated or cloner.has_mutated
        cloning_iter += 1

    with open(args.dst_impl_path, "w") if args.dst_impl_path else None as stream:
        info("I 'll dump the transformed tree to %s", args.dst_impl_path)
        ToPython(stream=stream).visit_topdown(tree)


if __name__ == "__main__":
    main()
