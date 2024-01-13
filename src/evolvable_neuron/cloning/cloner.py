from logging import debug

import numpy as np
from lark import Token, Transformer, Tree
from lark.load_grammar import Grammar


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

    def __set_grammar_objs(self, grammar: Grammar) -> None:
        self.rules = {str(rule[0]): rule[2] for rule in grammar.rule_defs}
        self.terms = {term[0]: term[1][0] for term in grammar.term_defs}
        self.expr_options = {
            expr: [child.children[-1].name for child in self.rules[expr].children]
            for expr in ("b_expr", "s_expr", "v_expr")
        }

    def reset(self) -> None:
        self._pending_assigment = {"b_assign": None, "s_assign": None, "v_assign": None}
        self._assigned_varnames = {"b_assign": [], "s_assign": [], "v_assign": []}
        self._new_assignments: list[str] = []
        self._curr_expr_curr_depth = 0
        self._curr_expr_max_depth = 0
        self.has_mutated = False

    def __default__(self, data, children, meta):
        def _possibly_mutate() -> Tree | None:
            for expr_type, options in self.expr_options.items():
                if data in options:
                    if self.rng.random() < self.mutation_rate:
                        debug("Mutating...")
                        self._curr_expr_max_depth = 0
                        expr = self.new_expr(expr_type)
                        debug("The new expression has a depth of %s", self._curr_expr_max_depth)
                    else:
                        expr = None
                    return expr
            return None

        if self.mutation_rate > 0:
            if data == "assign":
                # No reason to be doing this if ``self.mutation_rate == 0``
                self._pending_assigment[children[0].data] = children[0].children[0].children
            else:
                if (possibly_mutated_expr := _possibly_mutate()) is not None:
                    self.has_mutated = True
                    return possibly_mutated_expr

        return Tree(data, children, meta)

    def start(self, tree):
        return Tree(data="start", children=tree[:3] + self._new_assignments + tree[3:])

    def nl(self, tree):
        for assignment_type, varname in self._pending_assigment.items():
            if varname is not None:
                self._assigned_varnames[assignment_type].append(varname)
                self._pending_assigment[assignment_type] = None
                break

        return Tree(Token("RULE", "nl"), [])

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
                expr = getattr(self, "new_" + self.rng.choice(expr_options))()
            except Cloner.NoVariable:
                pass
            else:
                break

        if self.rng.random() < self.mutation_rate:
            # TODO: put `expr` in an assigment
            pass

        self._curr_expr_curr_depth -= 1
        return expr

    def new_b2b_expr(self) -> Tree:
        # b2b_op b_expr -> b2b_expr

        op_tree = Tree(Token("NOT_KW", "not "), [])

        while True:
            embedded_expr = self.new_expr("b_expr")
            if self.allow_all or embedded_expr.data != "b2b_expr":
                break

        return Tree(data="b2b_expr", children=[op_tree, embedded_expr])

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
            ],
        )

    def new_b_par_expr(self) -> Tree:
        # lpar b_expr rpar -> b_par_expr

        while True:
            embedded_expr = self.new_expr("b_expr")
            if self.allow_all:
                break

            if embedded_expr.data == "b_par_expr":
                continue

            if isinstance(embedded_expr.data, Token) and embedded_expr.data.value == "b_varname":
                continue

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

        op = self.rng.choice(self.rules["ss2b_op"].children)
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

    def new_b_var_expr(self) -> Tree:
        # b_varname : is sep noun sep adjective
        # b_varname -> b_var_expr

        if not len(self._assigned_varnames["b_assign"]):
            raise Cloner.NoVariable()

        var = self.rng.choice(self._assigned_varnames["b_assign"]).tolist()
        return Tree(data=Token("RULE", "b_varname"), children=var)

    def new_b_const_expr(self) -> Tree:
        # bconst : IS_S0_ABOVE_S1
        # bconst -> b_const_expr

        const_tree = Tree(Token("IS_S0_ABOVE_S1", "s0 > s1"), [])
        return Tree(data="b_const_expr", children=[Tree(Token("RULE", "bconst"), [const_tree])])

    def new_s2s_expr(self) -> Tree:
        # s_expr : s2s_op s_expr -> s2s_expr

        op = self.rng.choice(self.rules["s2s_op"].children)
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

        op = self.rng.choice(self.rules["after_s2s_op"].children)
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

        op = self.rng.choice(self.rules["ss2s_op"].children)
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

        const = self.rng.choice(self.rules["sconst"].children)
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

    def new_s_var_expr(self) -> Tree:
        # s_varname : adjective sep noun
        # s_varname -> s_var_expr

        if not len(self._assigned_varnames["s_assign"]):
            raise Cloner.NoVariable()

        var = self.rng.choice(self._assigned_varnames["s_assign"]).tolist()
        return Tree(data=Token("RULE", "s_varname"), children=var)

    def new_sv2v_expr(self) -> Tree:
        # v_expr : s_expr sv2v_op v_expr -> sv2v_expr

        op = self.rng.choice(self.rules["sv2v_op"].children)
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

        op = self.rng.choice(self.rules["vv2v_op"].children)
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

        const = self.rng.choice(self.rules["vconst"].children)
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

    def new_v_var_expr(self) -> Tree:
        # v_varname : adjective sep noun plural
        # v_varname -> v_var_expr

        if not len(self._assigned_varnames["v_assign"]):
            raise Cloner.NoVariable()

        var = self.rng.choice(self._assigned_varnames["v_assign"]).tolist()
        return Tree(data=Token("RULE", "v_varname"), children=var)
