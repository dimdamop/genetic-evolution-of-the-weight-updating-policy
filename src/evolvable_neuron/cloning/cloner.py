from collections import deque
from logging import debug
from typing import Literal, Deque, List, Tuple

import numpy as np
from lark import Token, Transformer, Visitor, Tree
from lark.load_grammar import Grammar


AssignTypeT = Literal["b_assign", "s_assign", "v_assign"]
VarnamesT = dict[AssignTypeT: List[Tree]]


def _choice(rng, seq):
    return seq[rng.choice(len(seq))]


class VarnameGenerator(Visitor):

    def __init__(self, adjective_rules, noun_rules, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adj_rules = adjective_rules[:-1]
        self.adj_and = adjective_rules[-1]
        self.noun_rules = noun_rules
        self.reset()

    def reset(self) -> None:
        self.varnames: VarnamesT = {"b_assign": [], "s_assign": [], "v_assign": []}

    def assign(self, tree):
        self.varnames[str(tree.children[0].data)].append(tree.children[0].children[0])

    @classmethod
    def _are_the_same(cls, varname1: Tree, varname2: Tree) -> bool:
        return False

        varname1 = varname1[0]
        varname2 = varname2[0]

        if len(varname1).children != len(varname2).children:
            return False

        for child1, child2 in zip(varname1.children, varname2.children):
            if child1.data != child2.data:
                return False

        return True

    def generate_b_varname(self, rng) -> str:
        noun_rule = _choice(rng, self.noun_rules)
        adj_rule = _choice(rng, self.adj_rules)
        noun = str(noun_rule.children[0].children[0].children[0].children[0]).strip('"')
        adj = str(adj_rule.children[0].children[0].children[0].children[0]).strip('"')

        return Tree(
            Token("RULE", "b_varname"),
            [
                Tree(Token("RULE", "is"), []),
                Tree(Token("RULE", "sep"), []),
                Tree(noun, []),
                Tree(Token("RULE", "sep"), []),
                Tree(adj, []),
            ],
        )

    def generate_s_varname(self, rng) -> str:
        noun_rule = _choice(rng, self.noun_rules)
        adj_rule = _choice(rng, self.adj_rules)
        noun = str(noun_rule.children[0].children[0].children[0].children[0]).strip('"')
        adj = str(adj_rule.children[0].children[0].children[0].children[0]).strip('"')

        return Tree(
            Token("RULE", "s_varname"),
            [Tree(adj, []), Tree(Token("RULE", "sep"), []), Tree(noun, [])],
        )

    def generate_v_varname(self, rng) -> str:
        return Tree(
            Token("RULE", "v_varname"),
            [self.generate_s_varname(rng), Tree(Token("RULE", "plural"), [])]
        )

    def generate_unobserved_varname(self, vartype: AssignTypeT, rng) -> str:
        while True:
            varname = getattr(self, f"generate_{vartype[0]}_varname")(rng)
            is_observed = False
            for existing_varname in self.varnames[vartype]:
                if self._are_the_same(existing_varname, varname):
                    is_observed = True
                    break
            if not is_observed:
                return varname


class Cloner(Transformer):
    class NoVariable(Exception):
        pass

    def __init__(
        self,
        grammar: Grammar,
        mutation_rate: float = 0,
        new_expr_pull_factor: float | None = 0.05,
        seed: int | None = None,
        allow_all: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mutation_rate = mutation_rate
        self.new_expr_pull_factor = new_expr_pull_factor or 0
        self.rng = np.random.default_rng(seed)
        self.allow_all = allow_all
        self.__set_grammar_objs(grammar)
        self.varname_gen = VarnameGenerator(
            adjective_rules=self.rules["adjective"].children,
            noun_rules=self.rules["noun"].children,
        )

    def __set_grammar_objs(self, grammar: Grammar) -> None:
        self.rules = {str(rule[0]): rule[2] for rule in grammar.rule_defs}
        self.terms = {term[0]: term[1][0] for term in grammar.term_defs}
        self.expr_options = {
            expr: [child.children[-1].name for child in self.rules[expr].children]
            for expr in ("b_expr", "s_expr", "v_expr")
        }

    def reset(self) -> None:
        self._available_varnames: VarnamesT = {"b_assign": [], "s_assign": [], "v_assign": []}
        self._curr_new_assignments: Deque[Tree] = deque()
        self._all_assignments: List[Tree] = []
        self._curr_expr_curr_depth = 0
        self._curr_expr_max_depth = 0
        self.has_mutated = False
        self.varname_gen.reset()

    def transform(self, tree):
        self.reset()
        self.varname_gen.visit(Tree(data=tree.data, children=tree.children))
        return super().transform(tree)

    def __default__(self, data, children, meta):
        def _possibly_new_expr() -> Tree | None:
            for expr_type, options in self.expr_options.items():
                if data in options:
                    if self.rng.random() < self.mutation_rate:
                        debug("Generating a new expression...")
                        self._curr_expr_max_depth = 0
                        expr = self.new_expr(expr_type)
                        debug("The generated expression is %s ops deep", self._curr_expr_max_depth)
                    else:
                        expr = None
                    return expr
            return None

        if self.mutation_rate > 0:
            if (new_expr := _possibly_new_expr()) is not None:
                self.has_mutated = True
                return new_expr

        return Tree(data, children, meta)

    def start(self, children):
        all_assignments = self._all_assignments + list(self._curr_new_assignments)
        return Tree(data="start", children=children[:3] + all_assignments + [children[-1]])

    def assign(self, children):
        if len(children) != 1:
            raise AssertionError("Unexpected structure for 'assign'")

        nested_assignment = Tree(Token("RULE", "assign"), children=children)
        assignment = Tree(
            data=Token("RULE", "indented_assign"),
            children=[
                Tree(Token("RULE", "tab"), []), nested_assignment, Tree(Token("RULE", "nl"), [])
            ]
        )

        # make this variable available to the next statements
        varname = children[0]
        self._available_varnames[str(varname.data)].append(varname.children[0])

        # record this assignment and any other ones created while processing it
        while True:
            try:
                # order is important here: assignments have to be added after their dependencies
                self._all_assignments.append(self._curr_new_assignments.popleft())
            except IndexError:
                self._all_assignments.append(assignment)
                break

        return nested_assignment

    def new_expr(self, expr_type) -> Tree:
        self._curr_expr_curr_depth += 1

        if self._curr_expr_max_depth < self._curr_expr_curr_depth:
            self._curr_expr_max_depth = self._curr_expr_curr_depth

        expr_options = self.expr_options[expr_type]

        if self.rng.random() > np.exp(-self.new_expr_pull_factor * self._curr_expr_max_depth):
            expr_options = [
                opt
                for opt in expr_options
                if opt.endswith("_const_expr") or opt.endswith("_var_expr")
            ]

        while True:
            try:
                expr = getattr(self, "new_" + _choice(self.rng, expr_options))()
            except Cloner.NoVariable:
                pass
            else:
                break

        if (
            (
                self.allow_all
                or not (
                    str(expr.data).endswith("_const_expr")
                    or str(expr.data).endswith("_varname")
                )
            )
            and self.rng.random() < self.mutation_rate
        ):
            expr = self.place_expr_in_assignment(expr, expr_type)

        self._curr_expr_curr_depth -= 1
        return expr

    def place_expr_in_assignment(self, expr: Tree, expr_type: str) -> Tree:
        expr_id: Literal["b", "s", "v"] = expr_type[0]
        assignment_type: AssignTypeT = expr_id + "_assign"
        varname = self.varname_gen.generate_unobserved_varname(assignment_type, self.rng)
        assignment = getattr(self, f"new_{assignment_type}")(varname, expr)
        self._available_varnames[assignment_type].append(varname)
        self._curr_new_assignments.append(assignment)
        debug("Placing an expression in an assignment with varname %s...", varname)
        return getattr(self, f"new_{expr_id}_var_expr")(varname)

    def new_b_assign(self, varname, expr) -> Tree:
        rhs = Tree(
            data=Token("RULE", "b_assign"),
            children=[varname, Tree(Token("RULE", "assign_kw"), []), expr],
        )

        return Tree(
            data=Token("RULE", "indented_assign"),
            children=
            [
                Tree(Token("RULE", "tab"), []),
                Tree(Token("RULE", "assign"), [rhs,]),
                Tree(Token("RULE", "nl"), [])
            ]
        )


    def new_s_assign(self, varname, expr) -> Tree:
        rhs = Tree(
            data=Token("RULE", "s_assign"),
            children=[varname, Tree(Token("RULE", "assign_kw"), []), expr],
        )

        return Tree(
            data=Token("RULE", "indented_assign"),
            children=
            [
                Tree(Token("RULE", "tab"), []),
                Tree(Token("RULE", "assign"), [rhs,]),
                Tree(Token("RULE", "nl"), [])
            ]
        )

    def new_v_assign(self, varname, expr) -> Tree:
        rhs = Tree(
            data=Token("RULE", "v_assign"),
            children=[varname, Tree(Token("RULE", "assign_kw"), []), expr],
        )

        return Tree(
            data=Token("RULE", "indented_assign"),
            children=
            [
                Tree(Token("RULE", "tab"), []),
                Tree(Token("RULE", "assign"), [rhs,]),
                Tree(Token("RULE", "nl"), [])
            ]
        )

    def new_b2b_expr(self) -> Tree:
        # b2b_op b_expr -> b2b_expr

        op_tree = Tree(Token("NOT_KW", "not "), [])

        while True:
            embedded_expr = self.new_expr("b_expr")
            # if ``not self.allow_all``, prohibit ``not not`` expressions
            if self.allow_all or embedded_expr.data != "b2b_expr":
                break

        return Tree(data="b2b_expr", children=[op_tree, embedded_expr])

    def new_bb2b_expr(self) -> Tree:
        # b_expr bb2b_op b_expr -> bb2b_expr

        op = _choice(self.rng, self.rules["bb2b_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="bb2b_expr",
            children=[
                self.new_expr("b_expr"),
                Tree(Token("RULE", "bb2b_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("b_expr"),
            ],
        )

    def new_b_par_expr(self) -> Tree:
        # lpar b_expr rpar -> b_par_expr

        while True:
            embedded_expr = self.new_expr("b_expr")
            if self.allow_all:
                break

            # if ``not self.allow_all``, prohibit ``(( ... ))`` expressions
            if embedded_expr.data == "b_par_expr":
                continue

            # if ``not self.allow_all``, prohibit ``(b_varname)`` expressions
            if isinstance(embedded_expr.data, Token) and embedded_expr.data.value == "b_varname":
                continue

            # if ``not self.allow_all``, prohibit ``(b_const_expr)`` expressions
            if embedded_expr.data == "b_const_expr":
                continue

            break

        return Tree(
            data="b_par_expr",
            children=[
                Tree(Token("RULE", "lpar"), []),
                embedded_expr,
                Tree(Token("RULE", "rpar"), []),
            ],
        )

    def new_ss2b_expr(self) -> Tree:
        # s_expr ss2b_op s_expr -> ss2b_expr

        op = _choice(self.rng, self.rules["ss2b_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        op_tree = Tree(Token("RULE", "ss2b_op"), [Token(op_name, op_str.strip('"'))])
        static_consts = "0", "1", "2"

        while True:
            left_expr = self.new_expr("s_expr")
            right_expr = self.new_expr("s_expr")

            is_statically_known = [
                expr.data == "s_const_expr" and expr.children[0].children[0].value in static_consts
                for expr in (left_expr, right_expr)
            ]

            if self.allow_all or not all(is_statically_known):
                break

        return Tree(data="ss2b_expr", children=[left_expr, op_tree, right_expr])

    def new_b_var_expr(self, varname: List[Tree] | None = None) -> Tree:
        # b_varname : is sep noun sep adjective
        # b_varname -> b_var_expr

        if varname is None:
            if not len(self._available_varnames["b_assign"]):
                raise Cloner.NoVariable()

            varname = _choice(self.rng, self._available_varnames["b_assign"])

        return Tree(data=Token("RULE", "b_varname"), children=[varname])

    def new_b_const_expr(self) -> Tree:
        # bconst : IS_S0_ABOVE_S1
        # bconst -> b_const_expr

        const_tree = Tree(Token("IS_S0_ABOVE_S1", "s0 > s1"), [])
        return Tree(data="b_const_expr", children=[Tree(Token("RULE", "bconst"), [const_tree])])

    def new_s2s_expr(self) -> Tree:
        # s_expr : s2s_op s_expr -> s2s_expr

        op = _choice(self.rng, self.rules["s2s_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        op_tree = (Tree(Token("RULE", "s2s_op"), [Token(op_name, op_str.strip('"'))]),)

        while True:
            embedded_expr = self.new_expr("s_expr")

            if (
                self.allow_all
                or embedded_expr.data != "s2s_op"
                # For both of the supported s2s operators `f` we have ``f(f(x)) = x``
                or op_name != embedded_expr.children[0].name
            ):
                break

        return Tree(data="s2s_expr", children=[op_tree, embedded_expr])

    def new_after_s2s_expr(self) -> Tree:
        # s_expr after_s2s_op -> after_s2s_expr

        op = _choice(self.rng, self.rules["after_s2s_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="after_s2s_expr",
            children=[
                self.new_expr("s_expr"),
                Tree(Token("RULE", "after_s2s_op"), [Token(op_name, op_str.strip('"'))]),
            ],
        )

    def new_ss2s_expr(self) -> Tree:
        # s_expr ss2s_op s_expr -> ss2s_expr

        op = _choice(self.rng, self.rules["ss2s_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        op_tree = Tree(Token("RULE", "ss2s_op"), [Token(op_name, op_str.strip('"'))])
        static_consts = "0", "1", "2"

        while True:
            left_expr = self.new_expr("s_expr")
            right_expr = self.new_expr("s_expr")

            if self.allow_all:
                break

            is_statically_known = [
                expr.data == "s_const_expr" and expr.children[0].children[0].value in static_consts
                for expr in (left_expr, right_expr)
            ]

            if all(is_statically_known):
                continue

            is_zero = [
                expr.data == "s_const_expr" and expr.children[0].children[0].value == "0"
                for expr in (left_expr, right_expr)
            ]

            if any(is_zero):
                continue

            is_one = [
                expr.data == "s_const_expr" and expr.children[0].children[0].value == "1"
                for expr in (left_expr, right_expr)
            ]

            if any(is_one) and op_name in ("MUL_KW", "DIV_KW"):
                continue

            break

        return Tree(data="s2s_expr", children=[left_expr, op_tree, right_expr])

    def new_s_par_expr(self) -> Tree:
        # lpar s_expr rpar -> s_par_expr

        while True:
            embedded_expr = self.new_expr("s_expr")

            if self.allow_all:
                break

            if embedded_expr.data == "s_par_expr":
                continue

            if isinstance(embedded_expr.data, Token) and embedded_expr.data.value == "s_varname":
                continue

            if embedded_expr.data == "s_const_expr":
                continue

            break

        return Tree(
            data="s_par_expr",
            children=[
                Tree(Token("RULE", "lpar"), []),
                embedded_expr,
                Tree(Token("RULE", "rpar"), []),
            ],
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
            ],
        )

    def new_vmean_expr(self) -> Tree:
        return Tree(
            data="vmean_expr",
            children=[
                Tree(Token("RULE", "mean"), []),
                Tree(Token("RULE", "lpar"), []),
                self.new_expr("v_expr"),
                Tree(Token("RULE", "rpar"), []),
            ],
        )

    def new_s_const_expr(self) -> Tree:
        # sconst : ZERO | ONE | TWO | B | DEPTH | S0 | S1
        # sconst -> s_const_expr

        const = _choice(self.rng, self.rules["sconst"].children)
        const_name = const.children[0].children[0].name
        const_str = self.terms[const_name].children[0].children[0].children[0].children[0].value

        return Tree(
            data="s_const_expr",
            children=[Tree(Token("RULE", "sconst"), [Token(const_name, const_str.strip('"'))])],
        )

    def new_s_ifelse_expr(self) -> Tree:
        # s_expr if b_expr else s_expr -> s_ifelse_expr

        return Tree(
            data="s_ifelse_expr",
            children=[
                self.new_expr("s_expr"),
                Tree(Token("RULE", "if"), []),
                self.new_expr("b_expr"),
                Tree(Token("RULE", "else"), []),
                self.new_expr("s_expr"),
            ],
        )

    def new_s_var_expr(self, varname: List[Tree] | None = None) -> Tree:
        # s_varname : adjective sep noun
        # s_varname -> s_var_expr

        if varname is None:
            if not len(self._available_varnames["s_assign"]):
                raise Cloner.NoVariable()

            varname = _choice(self.rng, self._available_varnames["s_assign"])

        return Tree(data=Token("RULE", "s_varname"), children=[varname])

    def new_sv2v_expr(self) -> Tree:
        # v_expr : s_expr sv2v_op v_expr -> sv2v_expr

        op = _choice(self.rng, self.rules["sv2v_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        op_tree = Tree(Token("RULE", "sv2v_op"), [Token(op_name, op_str.strip('"'))])

        while True:
            s_expr = self.new_expr("s_expr")
            v_expr = self.new_expr("v_expr")

            if self.allow_all:
                break

            if s_expr.data == "s_const_expr" and s_expr.children[0].children[0].value == "0":
                continue

            if (
                s_expr.data == "s_const_expr"
                and s_expr.children[0].children[0].value == "1"
                and op_name == "MUL_KW"
            ):
                continue

            break

        return Tree(data="sv2v_expr", children=[s_expr, op_tree, v_expr])

    def new_vv2v_expr(self) -> Tree:
        # v_expr vv2v_op v_expr -> vv2v_expr

        op = _choice(self.rng, self.rules["vv2v_op"].children)
        op_name = op.children[0].children[0].name
        op_str = self.terms[op_name].children[0].children[0].children[0].children[0].value
        return Tree(
            data="vv2v_expr",
            children=[
                self.new_expr("v_expr"),
                Tree(Token("RULE", "vv2v_op"), [Token(op_name, op_str.strip('"'))]),
                self.new_expr("v_expr"),
            ],
        )

    def new_v_const_expr(self) -> Tree:
        # vconst : W_VEC | INPUTS
        # vconst -> v_const_expr

        const = _choice(self.rng, self.rules["vconst"].children)
        const_name = const.children[0].children[0].name
        const_str = self.terms[const_name].children[0].children[0].children[0].children[0].value

        return Tree(
            data="v_const_expr",
            children=[Tree(Token("RULE", "vconst"), [Token(const_name, const_str.strip('"'))])],
        )

    def new_v_ifelse_expr(self) -> Tree:
        # v_expr if b_expr else v_expr -> v_ifelse_expr

        return Tree(
            data="v_ifelse_expr",
            children=[
                self.new_expr("v_expr"),
                Tree(Token("RULE", "if"), []),
                self.new_expr("b_expr"),
                Tree(Token("RULE", "else"), []),
                self.new_expr("v_expr"),
            ],
        )

    def new_v_var_expr(self, varname: List[Tree] | None = None) -> Tree:
        # v_varname : adjective sep noun plural
        # v_varname -> v_var_expr

        if varname is None:
            if not len(self._available_varnames["v_assign"]):
                raise Cloner.NoVariable()

            varname = _choice(self.rng, self._available_varnames["v_assign"])

        return Tree(data=Token("RULE", "v_varname"), children=[varname])
