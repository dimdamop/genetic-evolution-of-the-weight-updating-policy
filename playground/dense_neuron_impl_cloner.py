from argparse import ArgumentParser, Namespace
from pathlib import Path
from logging import info
from sys import stdout
import numpy as np
from lark import Lark, Tree, Visitor, Token, Transformer
from lark.load_grammar import Grammar


class ToPython(Visitor):
    Mapper = {
        "nl": "\n",
        "tab": "    ",
        "sep": "_",
        "assign_kw": " = ",
        "lpar": "(",
        "rpar": ")",
        "dot": "jnp.dot",
        "mean": "jnp.mean",
        "comma": ", ",
        "ret": "return ",
        "if": " if ",
        "else": " else ",
        "plural": "s",
        "grammar_and": "_and_",
    }

    def __init__(self, stream=stdout, *args, **kwargs):
        self.stream = stream
        super().__init__(*args, **kwargs)

    def __default__(self, tree):

        if len(tree.children) == 0:
            self.stream.write(ToPython.Mapper.get(tree.data, tree.data))
            return
        
        if all(isinstance(child, Token) for child in tree.children):
            for child in tree.children:
                self.stream.write(str(child))

    def start(self, tree):
        for child in tree.children[:3]:
            self.stream.write(str(child))


class Cloner(Transformer):

    class NoVariable(Exception):
        pass

    def __init__(
        self,
        grammar: Grammar,
        mutation_rate: float = 0,
        seed: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.__set_grammar_objs(grammar)

    def __set_grammar_objs(self, grammar: Grammar) -> None:
        self.rules = {str(rule[0]): rule[2] for rule in grammar.rule_defs}
        self.terms = {term[0]: term[1][0] for term in grammar.term_defs}
        self.expr_options = {
            expr: [child.children[-1].name for child in self.rules[expr].children]
            for expr in ("b_expr", "s_expr", "v_expr")
        }

    def reset(self) -> None:
        self.num_mutations: int = 0
        self._varnames = {"b_varname": [], "s_varname": [], "v_varname": []}
        self._new_assignments: list[str] = []

    def start(self, tree):
        return Tree(data="start", children=tree[:3] + self._new_assignments + tree[3:])

    def __default__(self, data, children, meta):
        if self.mutation_rate > 0:
            if self.rng.random() < self.mutation_rate:
                for expr_type, options in self.expr_options.items():
                    if data in options:
                        return self.new_expr(expr_type)

            if data in ("b_varname", "s_varname", "v_varname"):
                self._varnames[data].append(children)

        return Tree(data, children, meta)

    def new_expr(self, expr_type) -> Tree:
        while True:
            try:
                expr = getattr(self, "new_" + self.rng.choice(self.expr_options[expr_type]))()
                break
            except Cloner.NoVariable:
                pass

        if self.rng.random() >= self.mutation_rate:
            return expr

        # TODO: put `expr` in an assigment
        return expr

    def new_b2b_expr(self) -> Tree:
        # b2b_op b_expr -> b2b_expr
        return Tree(
            data="b_expr",
            children=[
                Tree(Token("NOT_KW", "not "), []),
                self.new_expr("b_expr"),
            ] 
        )

    def new_bb2b_expr(self) -> Tree:
        # b_expr bb2b_op b_expr -> bb2b_expr

        op = self.rng.choice(self.rules["bb2b_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="bb2b_expr",
            children=[
                self.new_expr("b_expr"),
                Tree(Token("RULE", "bb2b_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("b_expr"),
            ] 
        )

    def new_bpar_expr(self) -> Tree:
        # lpar b_expr rpar      -> bpar_expr
        return Tree(
            data="bpar_expr",
            children=[
                Tree(Token("RULE", "lpar"), []),
                self.new_expr("b_expr"),
                Tree(Token("RULE", "rpar"), []),
            ]
        )

    def new_ss2b_expr(self) -> Tree:
        # s_expr ss2b_op s_expr -> ss2b_expr

        op = self.rng.choice(self.rules["ss2b_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="ss2b_expr",
            children=[
                self.new_expr("s_expr"),
                Tree(Token("RULE", "ss2b_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("s_expr"),
            ] 
        )

    def new_b_var_expr(self) -> Tree:
        # b_varname : is sep noun sep adjective
        # b_varname -> b_var_expr
        if not len(self._varnames["b_varname"]):
            raise Cloner.NoVariable()

        return Tree(
            data=Token("RULE", "b_varname"),
            children=self.rng.choice(self._varnames['b_varname']).tolist()
        )

    def new_s2s_expr(self) -> Tree:
        # s_expr : s2s_op s_expr -> s2s_expr

        op = self.rng.choice(self.rules["s2s_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="s2s_expr",
            children=[
                Tree(Token("RULE", "s2s_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("s_expr"),
            ] 
        )

    def new_ss2s_expr(self) -> Tree:
        # s_expr ss2s_op s_expr -> ss2s_expr

        op = self.rng.choice(self.rules["ss2s_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="s2s_expr",
            children=[
                self.new_expr("s_expr"),
                Tree(Token("RULE", "ss2s_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("s_expr"),
            ] 
        )

    def new_s_par_expr(self) -> Tree:
        # lpar s_expr rpar -> s_par_expr
        return Tree(
            data="spar_expr",
            children=[
                Tree(Token("RULE", "lpar"), []),
                self.new_expr("s_expr"),
                Tree(Token("RULE", "rpar"), []),
            ]
        )

    def new_vprod_expr(self) -> Tree:
        # dot lpar v_expr comma v_expr rpar -> vprod_expr
        return Tree(
            data="vprod_expr",
            children=[
                Tree(Token("RULE", "dot"), []),
                Tree(Token("RULE", "lpar"), []),
                self.new_expr("v_expr"),
                Tree(Token("RULE", "comma"), []),
                self.new_expr("v_expr"),
                Tree(Token("RULE", "rpar"), []),
            ]
        )

    def new_vmean_expr(self) -> Tree:
        return Tree(
            data="vmean_expr",
            children=[
                Tree(Token("RULE", "mean"), []),
                Tree(Token("RULE", "lpar"), []),
                self.new_expr("v_expr"),
                Tree(Token("RULE", "rpar"), []),
            ]
        )

    def new_s_const_expr(self) -> Tree:
        # sconst : ZERO | ONE | TWO | B | DEPTH | S0 | S1
        # sconst -> s_const_expr
        const = self.rng.choice(self.rules["sconst"].children)
        const_name = const.children[0].children[0].name
        const_str = self.terms[const_name].children[0].children[0].children[0].children[0].value

        return Tree(
            data="s_const_expr",
            children=[Tree(Token("RULE", "sconst"), [Token(const_name, const_str.strip('"'))])],
        )

    def new_s_ifelse(self) -> Tree:
        # s_expr if b_expr else s_expr -> s_ifelse
        return Tree(
            data="s_ifelse",
            children=[
                self.new_expr("s_expr"),
                Tree(Token("RULE", "if"), []),
                self.new_expr("b_expr"),
                Tree(Token("RULE", "else"), []),
                self.new_expr("s_expr"),
            ] 
        )

    def new_s_var_expr(self) -> Tree:
        # s_varname : adjective sep noun
        # s_varname -> s_var_expr
        if not len(self._varnames["s_varname"]):
            raise Cloner.NoVariable()

        return Tree(
            data=Token("RULE", "s_varname"),
            children=self.rng.choice(self._varnames['s_varname']).tolist()
        )

    def new_sv2v_expr(self) -> Tree:
        # v_expr : s_expr sv2v_op v_expr -> sv2v_expr
        op = self.rng.choice(self.rules["sv2v_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="sv2v_expr",
            children=[
                self.new_expr("s_expr"),
                Tree(Token("RULE", "sv2v_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("v_expr"),
            ] 
        )

    def new_vv2v_expr(self) -> Tree:
        # v_expr vv2v_op v_expr -> vv2v_expr
        op = self.rng.choice(self.rules["vv2v_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="vv2v_expr",
            children=[
                self.new_expr("v_expr"),
                Tree(Token("RULE", "vv2v_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("v_expr"),
            ] 
        )

    def new_v_const_expr(self) -> Tree:
        # vconst : W_VEC | INPUTS
        # vconst -> v_const_expr
        const = self.rng.choice(self.rules["vconst"].children)
        const_name = const.children[0].children[0].name
        const_str = self.terms[const_name].children[0].children[0].children[0].children[0].value

        return Tree(
            data="v_const_expr",
            children=[Tree(Token("RULE", "vconst"), [Token(const_name, const_str.strip('"'))])],
        )

    def new_v_ifelse(self) -> Tree:
        # v_expr if b_expr else v_expr -> v_ifelse
        return Tree(
            data="v_ifelse",
            children=[
                self.new_expr("v_expr"),
                Tree(Token("RULE", "if"), []),
                self.new_expr("b_expr"),
                Tree(Token("RULE", "else"), []),
                self.new_expr("v_expr"),
            ] 
        )

    def new_v_var_expr(self) -> Tree:
        # v_varname : adjective sep noun plural
        # v_varname -> v_var_expr
        if not len(self._varnames["v_varname"]):
            raise Cloner.NoVariable()

        return Tree(
            data=Token("RULE", "v_varname"),
            children=self.rng.choice(self._varnames['v_varname']).tolist()
        )


def get_args() -> Namespace:
    argparser = ArgumentParser()
    argparser.add_argument("--src-impl-path")
    argparser.add_argument("--dst-impl-path")
    argparser.add_argument("--mutation-rate", type=float)
    argparser.add_argument("--cloning-iters", type=int)
    argparser.add_argument("--random-seed", type=int)

    args = argparser.parse_args()
    args.cloning_iters = args.cloning_iters or 1
    args.mutation_rate = args.mutation_rate or 0
    return args


def main():

    args = get_args()
    parser = Lark.open(Path(__file__).parent / "dense_neuron_impl.lark")

    with open(args.src_impl_path) as stream:
        info("I 'll parse %s according to the %s grammar", args.src_impl_path, parser.source_path)
        tree = parser.parse(stream.read())

    cloner = Cloner(grammar=parser.grammar, mutation_rate=args.mutation_rate, seed=args.random_seed)
    info("I 'll clone with a mutation rate of %f. Seed is %s", args.mutation_rate, args.random_seed)

    for i in range(1, args.cloning_iters + 1):
        info("I am on the cloning iteration %d/%d", i, args.cloning_iters)
        cloner.reset()
        tree = cloner.transform(tree)

    with open(args.dst_impl_path, "w") if args.dst_impl_path else None as stream:
        info("I 'll dump the transformed tree to %s", args.dst_impl_path)
        ToPython(stream=stream).visit_topdown(tree)


if __name__ == "__main__":
    main()
