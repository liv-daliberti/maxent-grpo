"""Fail-closed route extraction from numeric equation chains.

The 0.5B base policy rarely follows a separate JSON or RPN trace interface, but
it frequently emits arithmetic equations in its ordinary derivation. This
module recognizes only a small expression grammar, grounds every numeric leaf
in the problem or an earlier checked equality, and re-executes every accepted
operation. It does not execute Python, parse prose, or ask a model to judge a
strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Iterable

import sympy

from .math_route import MathRouteValidation, problem_number_inventory


MATH_EQUATION_ROUTE_VERSION = "math-equation-route-v3"
_MAX_EXPRESSION_CHARS = 1_024
_MAX_EXPRESSION_TOKENS = 96
_MAX_ROUTE_STEPS = 40
_MATH_SPAN_RE = re.compile(
    r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$",
    re.DOTALL,
)
_ROUTE_BLOCK_RE = re.compile(r"<route>.*?</route>", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"\s*(?:(-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*)?)"
    r"|([A-Za-z][A-Za-z0-9_]*)|([()+\-*/^!,=]))"
)
_NUMBER_FULL_RE = re.compile(
    r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*)?"
)
_COMMUTATIVE = frozenset({"add", "mul", "gcd", "lcm", "min", "max"})


class EquationRouteError(ValueError):
    """Raised when a candidate equation leaves the restricted grammar."""


@dataclass(frozen=True)
class _Node:
    kind: str
    value: str | Fraction | None = None
    children: tuple["_Node", ...] = ()


@dataclass(frozen=True)
class _Route:
    value: sympy.Expr
    signature: str
    operations: tuple[str, ...]
    step_count: int
    source_count: int

    @property
    def computed(self) -> bool:
        return any(operation != "source" for operation in self.operations)


def _fraction(text: str) -> Fraction:
    compact = str(text).replace(",", "")
    if "/" in compact:
        numerator, denominator = compact.split("/", 1)
        if int(denominator) == 0:
            raise EquationRouteError("zero denominator")
        return Fraction(int(numerator), int(denominator))
    return Fraction(compact)


def _bounded(value: sympy.Expr) -> sympy.Expr:
    simplified = sympy.simplify(value)
    if simplified.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo, sympy.I):
        raise EquationRouteError("non-real or non-finite equation value")
    if len(str(simplified)) > 1_024 or int(sympy.count_ops(simplified)) > 128:
        raise EquationRouteError("equation value is too large")
    for atom in simplified.atoms(sympy.Integer):
        if abs(int(atom)).bit_length() > 512:
            raise EquationRouteError("equation integer is too large")
    return simplified


def _braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise EquationRouteError("LaTeX command lacks a braced argument")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise EquationRouteError("unbalanced LaTeX braces")


def _expand_latex_commands(text: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        command = None
        for candidate in (r"\dfrac", r"\frac", r"\binom", r"\sqrt", r"\boxed"):
            if text.startswith(candidate, index):
                command = candidate
                break
        if command is None:
            output.append(text[index])
            index += 1
            continue
        cursor = index + len(command)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        first, cursor = _braced(text, cursor)
        first = _expand_latex_commands(first)
        if command in {r"\frac", r"\dfrac", r"\binom"}:
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            second, cursor = _braced(text, cursor)
            second = _expand_latex_commands(second)
            if command == r"\binom":
                output.append(f"choose(({first}),({second}))")
            else:
                output.append(f"(({first})/({second}))")
        elif command == r"\sqrt":
            output.append(f"sqrt(({first}))")
        else:
            output.append(f"({first})")
        index = cursor
    return "".join(output)


def _strip_text_commands(text: str) -> str:
    result = text
    for command in (r"\text", r"\mathrm", r"\mathbf", r"\operatorname"):
        while True:
            start = result.find(command)
            if start < 0:
                break
            cursor = start + len(command)
            while cursor < len(result) and result[cursor].isspace():
                cursor += 1
            try:
                content, stop = _braced(result, cursor)
            except EquationRouteError:
                return ""
            replacement = content if command == r"\operatorname" else ""
            result = result[:start] + replacement + result[stop:]
    return result


def _normalize_expression(text: str) -> str:
    if len(str(text)) > _MAX_EXPRESSION_CHARS:
        raise EquationRouteError("equation span is too long")
    normalized = _strip_text_commands(str(text))
    normalized = _expand_latex_commands(normalized)
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\cdot", "*").replace(r"\times", "*")
    normalized = normalized.replace(r"\div", "/")
    normalized = normalized.replace(r"\pi", "pi")
    normalized = normalized.replace(r"\%", "/100")
    normalized = normalized.replace(r"\implies", "=")
    normalized = normalized.replace(r"\Rightarrow", "=")
    normalized = normalized.replace(r"\quad", " ")
    normalized = normalized.replace(r"\,", " ")
    normalized = normalized.replace(r"\!", " ")
    normalized = normalized.replace(r"\\", " ")
    normalized = normalized.replace("−", "-").replace("×", "*").replace("÷", "/")
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = re.sub(r"_\(([^()]*)\)", r"_\1", normalized)
    normalized = re.sub(r"_([A-Za-z0-9]+)", r"\1", normalized)
    normalized = normalized.replace("&", " ")
    normalized = normalized.strip().strip(".,:;")
    if normalized.startswith("|") and normalized.endswith("|"):
        normalized = f"abs({normalized[1:-1]})"
    return normalized


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_expression(text)
    tokens: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        match = _TOKEN_RE.match(normalized, cursor)
        if match is None:
            if normalized[cursor:].strip():
                raise EquationRouteError("unsupported character in equation")
            break
        token = next(value for value in match.groups() if value is not None)
        tokens.append(token)
        cursor = match.end()
    if not tokens or len(tokens) > _MAX_EXPRESSION_TOKENS:
        raise EquationRouteError("equation has an invalid token count")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise EquationRouteError("unexpected end of expression")
        self.position += 1
        return token

    @staticmethod
    def _starts_atom(token: str | None) -> bool:
        return bool(
            token is not None
            and (
                _NUMBER_FULL_RE.fullmatch(token)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token)
                or token == "("
            )
        )

    def parse(self) -> _Node:
        node = self.parse_add()
        if self.peek() is not None:
            raise EquationRouteError("unconsumed expression token")
        return node

    def parse_add(self) -> _Node:
        node = self.parse_mul()
        while self.peek() in {"+", "-"}:
            operator = self.take()
            right = self.parse_mul()
            node = _Node("add" if operator == "+" else "sub", children=(node, right))
        return node

    def parse_mul(self) -> _Node:
        node = self.parse_power()
        while True:
            token = self.peek()
            if token in {"*", "/"}:
                operator = self.take()
                right = self.parse_power()
                node = _Node(
                    "mul" if operator == "*" else "div",
                    children=(node, right),
                )
            elif self._starts_atom(token):
                node = _Node("mul", children=(node, self.parse_power()))
            else:
                return node

    def parse_power(self) -> _Node:
        node = self.parse_unary()
        if self.peek() == "^":
            self.take()
            exponent = self.parse_power()
            exponent_value = _atomic_value(exponent)
            if exponent_value == 2:
                node = _Node("square", children=(node,))
            elif exponent_value == 3:
                node = _Node("cube", children=(node,))
            else:
                node = _Node("pow", children=(node, exponent))
        return node

    def parse_unary(self) -> _Node:
        if self.peek() == "+":
            self.take()
            return self.parse_unary()
        if self.peek() == "-":
            self.take()
            return _Node("neg", children=(self.parse_unary(),))
        node = self.parse_atom()
        while self.peek() == "!":
            self.take()
            node = _Node("factorial", children=(node,))
        return node

    def parse_atom(self) -> _Node:
        token = self.take()
        if _NUMBER_FULL_RE.fullmatch(token):
            return _Node("number", value=_fraction(token))
        if token == "(":
            node = self.parse_add()
            if self.take() != ")":
                raise EquationRouteError("unbalanced expression parentheses")
            return node
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token):
            if self.peek() != "(":
                return _Node("symbol", value=token)
            if token not in {
                "sqrt",
                "abs",
                "choose",
                "gcd",
                "lcm",
                "min",
                "max",
            }:
                # Treat f(x) as multiplication by an uninterpreted symbolic
                # function label only by rejecting it. Numeric sides of the
                # same equality chain remain independently usable.
                raise EquationRouteError("unsupported equation function")
            self.take()
            arguments = [self.parse_add()]
            while self.peek() == ",":
                self.take()
                arguments.append(self.parse_add())
            if self.take() != ")":
                raise EquationRouteError("unbalanced function arguments")
            expected = 1 if token in {"sqrt", "abs"} else 2
            if len(arguments) != expected:
                raise EquationRouteError("equation function has wrong arity")
            return _Node(token, children=tuple(arguments))
        raise EquationRouteError("unsupported equation atom")


def _parse(text: str) -> _Node:
    return _Parser(_tokenize(text)).parse()


def _route_key(route: _Route) -> tuple[int, int, str]:
    return (
        sum(operation != "source" for operation in route.operations),
        route.step_count,
        route.signature,
    )


def _store(known: dict[sympy.Expr, _Route], route: _Route) -> bool:
    value = _bounded(route.value)
    normalized = _Route(
        value=value,
        signature=route.signature,
        operations=route.operations,
        step_count=route.step_count,
        source_count=route.source_count,
    )
    previous = known.get(value)
    if previous is None or _route_key(normalized) < _route_key(previous):
        known[value] = normalized
        return True
    return False


def _signature(kind: str, children: Iterable[str]) -> str:
    parts = tuple(children)
    if kind in _COMMUTATIVE:
        parts = tuple(sorted(parts))
    return f"{kind}({','.join(parts)})"


def _execute(kind: str, arguments: tuple[sympy.Expr, ...]) -> sympy.Expr:
    if kind == "neg":
        value = -arguments[0]
    elif kind == "add":
        value = arguments[0] + arguments[1]
    elif kind == "sub":
        value = arguments[0] - arguments[1]
    elif kind == "mul":
        value = arguments[0] * arguments[1]
    elif kind == "div":
        if sympy.simplify(arguments[1]) == 0:
            raise EquationRouteError("division by zero")
        value = arguments[0] / arguments[1]
    elif kind == "pow":
        exponent = sympy.simplify(arguments[1])
        if exponent.is_Integer is not True or not -12 <= int(exponent) <= 12:
            raise EquationRouteError("equation exponent is outside [-12,12]")
        value = arguments[0] ** int(exponent)
    elif kind == "square":
        value = arguments[0] ** 2
    elif kind == "cube":
        value = arguments[0] ** 3
    elif kind == "sqrt":
        if arguments[0].is_nonnegative is not True:
            raise EquationRouteError("sqrt argument is not provably nonnegative")
        value = sympy.sqrt(arguments[0])
    elif kind == "abs":
        value = sympy.Abs(arguments[0])
    elif kind == "factorial":
        argument = sympy.simplify(arguments[0])
        if argument.is_Integer is not True or not 0 <= int(argument) <= 100:
            raise EquationRouteError("factorial argument is outside [0,100]")
        value = sympy.factorial(int(argument))
    elif kind == "choose":
        left, right = (sympy.simplify(argument) for argument in arguments)
        if (
            left.is_Integer is not True
            or right.is_Integer is not True
            or not 0 <= int(right) <= int(left) <= 10_000
        ):
            raise EquationRouteError("choose arguments are invalid")
        value = sympy.binomial(int(left), int(right))
    elif kind in {"gcd", "lcm"}:
        integers = []
        for argument in arguments:
            simplified = sympy.simplify(argument)
            if simplified.is_Integer is not True:
                raise EquationRouteError(f"{kind} argument is not an integer")
            integers.append(int(simplified))
        function = math.gcd if kind == "gcd" else math.lcm
        value = sympy.Integer(function(*integers))
    elif kind == "min":
        value = sympy.Min(*arguments)
    elif kind == "max":
        value = sympy.Max(*arguments)
    else:
        raise EquationRouteError("unsupported equation operation")
    return _bounded(value)


def _evaluate(
    node: _Node,
    known: dict[sympy.Expr, _Route],
    *,
    allow_symbols: bool,
) -> _Route:
    if node.kind == "number":
        assert isinstance(node.value, Fraction)
        value = sympy.Rational(node.value.numerator, node.value.denominator)
        route = known.get(value)
        if route is None:
            raise EquationRouteError("equation literal is not grounded")
        return route
    if node.kind == "symbol":
        assert isinstance(node.value, str)
        if node.value == "pi":
            route = known.get(sympy.pi)
            if route is None:
                raise EquationRouteError("pi is not grounded in the problem")
            return route
        if not allow_symbols:
            raise EquationRouteError("numeric equation contains a variable")
        return _Route(
            value=sympy.Symbol(node.value),
            signature="var",
            operations=(),
            step_count=1,
            source_count=0,
        )
    children = tuple(
        _evaluate(child, known, allow_symbols=allow_symbols)
        for child in node.children
    )
    value = _execute(node.kind, tuple(child.value for child in children))
    return _Route(
        value=value,
        signature=_signature(node.kind, (child.signature for child in children)),
        operations=tuple(
            operation for child in children for operation in child.operations
        )
        + (node.kind,),
        step_count=sum(child.step_count for child in children) + 1,
        source_count=sum(child.source_count for child in children),
    )


def _atomic_value(node: _Node) -> sympy.Expr | None:
    if node.kind == "number":
        assert isinstance(node.value, Fraction)
        return sympy.Rational(node.value.numerator, node.value.denominator)
    if node.kind == "neg" and len(node.children) == 1:
        child = _atomic_value(node.children[0])
        return -child if child is not None else None
    return None


def _math_spans(response: str) -> list[str]:
    clean = _ROUTE_BLOCK_RE.sub(" ", str(response))
    return [
        next(group for group in match.groups() if group is not None)
        for match in _MATH_SPAN_RE.finditer(clean)
    ]


def _equation_sides(span: str) -> list[_Node]:
    normalized = _normalize_expression(span)
    if "=" not in normalized:
        return []
    sides: list[_Node] = []
    for raw in normalized.split("="):
        candidate = raw.strip()
        if not candidate:
            continue
        try:
            sides.append(_parse(candidate))
        except EquationRouteError:
            continue
    return sides if len(sides) >= 2 else []


def _learn_numeric_equalities(
    sides: list[_Node],
    known: dict[sympy.Expr, _Route],
) -> bool:
    changed = False
    evaluable: list[_Route] = []
    for node in sides:
        try:
            route = _evaluate(node, known, allow_symbols=False)
        except EquationRouteError:
            continue
        if route.computed:
            evaluable.append(route)
    for route in evaluable:
        for node in sides:
            value = _atomic_value(node)
            if value is not None and _bounded(value - route.value) == 0:
                changed = _store(known, route) or changed
    return changed


def _learn_symbolic_solutions(
    sides: list[_Node],
    known: dict[sympy.Expr, _Route],
) -> bool:
    changed = False
    evaluated: list[_Route] = []
    for node in sides:
        try:
            evaluated.append(_evaluate(node, known, allow_symbols=True))
        except EquationRouteError:
            continue
    for left_index, left in enumerate(evaluated):
        for right in evaluated[left_index + 1 :]:
            equation = _bounded(left.value - right.value)
            symbols = sorted(equation.free_symbols, key=str)
            if len(symbols) != 1 or int(sympy.count_ops(equation)) > 32:
                continue
            try:
                solutions = sympy.solve(equation, symbols[0])
            except Exception:
                continue
            if not isinstance(solutions, list) or not 1 <= len(solutions) <= 2:
                continue
            relation_signature = _signature(
                "eq",
                (left.signature, right.signature),
            )
            relation_operations = left.operations + right.operations + ("solve",)
            for solution in solutions:
                solution = _bounded(solution)
                if solution.free_symbols or solution.is_real is not True:
                    continue
                changed = (
                    _store(
                        known,
                        _Route(
                            value=solution,
                            signature=_signature("solve", (relation_signature,)),
                            operations=relation_operations,
                            step_count=left.step_count + right.step_count + 1,
                            source_count=left.source_count + right.source_count,
                        ),
                    )
                    or changed
                )
    return changed


def _boxed_nodes(response: str) -> list[_Node]:
    nodes: list[_Node] = []
    marker = r"\boxed"
    cursor = 0
    while True:
        start = response.find(marker, cursor)
        if start < 0:
            break
        brace = start + len(marker)
        while brace < len(response) and response[brace].isspace():
            brace += 1
        try:
            content, cursor = _braced(response, brace)
            nodes.append(_parse(content))
        except EquationRouteError:
            cursor = max(brace + 1, start + len(marker))
    return nodes


def validate_math_equation_route_response(
    model_response: str,
    problem: str,
) -> MathRouteValidation | None:
    """Return a verified route extracted from checked equation transitions."""

    try:
        inventory = problem_number_inventory(problem)
        known: dict[sympy.Expr, _Route] = {}
        for value in inventory:
            expression = sympy.Rational(value.numerator, value.denominator)
            _store(
                known,
                _Route(
                    value=expression,
                    signature="input",
                    operations=("source",),
                    step_count=1,
                    source_count=1,
                ),
            )
        if re.search(r"\\pi\b|\bpi\b", str(problem), re.IGNORECASE):
            _store(
                known,
                _Route(
                    value=sympy.pi,
                    signature="input",
                    operations=("source",),
                    step_count=1,
                    source_count=1,
                ),
            )
        if not known:
            return None

        for span in _math_spans(model_response):
            sides = _equation_sides(span)
            if not sides:
                continue
            for _ in range(3):
                changed = _learn_numeric_equalities(sides, known)
                changed = _learn_symbolic_solutions(sides, known) or changed
                if not changed:
                    break

        for node in reversed(_boxed_nodes(model_response)):
            try:
                terminal = _evaluate(node, known, allow_symbols=False)
            except EquationRouteError:
                value = _atomic_value(node)
                terminal = known.get(_bounded(value)) if value is not None else None
            if (
                terminal is None
                or not terminal.computed
                or terminal.step_count > _MAX_ROUTE_STEPS
            ):
                continue
            return MathRouteValidation(
                route_signature=(
                    f"math-route:{MATH_EQUATION_ROUTE_VERSION}:"
                    f"{terminal.signature}"
                ),
                terminal_value=terminal.value,
                operations=terminal.operations,
                step_count=terminal.step_count,
                source_count=terminal.source_count,
            )
        return None
    except Exception:
        return None


def equation_formatting_variant(response: str) -> str:
    """Perturb equation whitespace without changing tokens or prose."""

    def perturb(match: re.Match[str]) -> str:
        groups = match.groups()
        content = next(group for group in groups if group is not None)
        changed = re.sub(r"\s*=\s*", "  =\n", content)
        whole = match.group(0)
        start = whole.find(content)
        return whole[:start] + changed + whole[start + len(content) :]

    return _MATH_SPAN_RE.sub(perturb, str(response))
