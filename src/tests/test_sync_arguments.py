"""Guard against drift between CLI flags, embed() API, and embedder args."""
import argparse
import ast
import inspect
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("src"))

from pepe import api
from pepe.parse_arguments import parse_arguments

CLI_ONLY = {"check_model"}
EMBED_ONLY = {"sequences"}


class _ParserCapture(Exception):
    pass


def _cli_dests():
    dests = set()

    def capture(self, args=None, namespace=None):
        dests.update(
            action.dest
            for action in self._actions
            if action.dest not in (None, "help", argparse.SUPPRESS)
        )
        raise _ParserCapture()

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            parse_arguments()
        except _ParserCapture:
            pass
    return dests


def _embed_params():
    return {name for name in inspect.signature(api.embed).parameters if name != "kwargs"}


def _args_dict_keys():
    tree = ast.parse(inspect.getsource(api.embed))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "args_dict":
                return {key.value for key in node.value.keys}
    raise RuntimeError("Could not find args_dict in pepe.api.embed")


def _kwargs_keys_in_args_dict():
    tree = ast.parse(inspect.getsource(api.embed))
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id == "kwargs"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
    return keys


def _args_attrs_from_class(cls):
    """Collect args.<attr> and getattr(args, '<attr>') via AST on the class body."""
    path = inspect.getfile(cls)
    with open(path, encoding="utf-8") as f:
        module_source = f.read()
    tree = ast.parse(module_source, filename=path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    )

    attrs = set()
    for node in ast.walk(class_node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            attrs.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attrs.add(node.args[1].value)
    return attrs


def _embedder_arg_attrs():
    """Attributes read from args in BaseEmbedder and CustomEmbedder."""
    from pepe.embedders import base_embedder, custom_embedder

    attrs = set()
    for cls in (base_embedder.BaseEmbedder, custom_embedder.CustomEmbedder):
        attrs.update(_args_attrs_from_class(cls))
    attrs.add("trust_remote_code")
    return attrs


class TestArgumentSync(unittest.TestCase):
    def test_cli_embed_and_args_dict_stay_in_sync(self):
        cli = _cli_dests()
        embed_params = _embed_params()
        args_dict = _args_dict_keys()
        embedder_attrs = _embedder_arg_attrs()

        cli_embedder = cli - CLI_ONLY
        embed_surface = (embed_params - EMBED_ONLY) | _kwargs_keys_in_args_dict()

        self.assertEqual(
            cli_embedder,
            args_dict,
            msg=(
                "CLI flags (minus check_model) must match embed() args_dict keys. "
                f"CLI only: {sorted(cli_embedder - args_dict)}; "
                f"args_dict only: {sorted(args_dict - cli_embedder)}"
            ),
        )
        self.assertEqual(
            embed_surface,
            args_dict,
            msg=(
                "embed() parameters (explicit + kwargs) must match args_dict keys. "
                f"embed only: {sorted(embed_surface - args_dict)}; "
                f"args_dict only: {sorted(args_dict - embed_surface)}"
            ),
        )
        missing = embedder_attrs - args_dict
        self.assertFalse(
            missing,
            msg=f"BaseEmbedder reads args not wired through embed(): {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main()
