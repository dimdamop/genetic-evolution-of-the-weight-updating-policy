from sys import stdout

from lark import Token, Visitor


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
        self._stream = stream
        super().__init__(*args, **kwargs)

    def __default__(self, tree):
        if len(tree.children) == 0:
            self._stream.write(ToPython.Mapper.get(tree.data, tree.data))
            return

        if all(isinstance(child, Token) for child in tree.children):
            for child in tree.children:
                self._stream.write(str(child))

    def start(self, tree):
        for child in tree.children[:3]:
            self._stream.write(str(child))
