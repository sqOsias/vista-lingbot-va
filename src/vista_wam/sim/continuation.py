"""Run a scoped RMBench suffix beyond native success without clearing its flag.

Inputs: a fresh environment and its original take_action method. Outputs:
instance-only driver replacing the success lock/early return with event logging.
Official check_success, stage updates, motion planners and source files remain
unchanged. This is a diagnostic driver, not official full-task evaluation.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import types


def install_continuation(env, output):
    """Bind exactly two AST changes; preserve the native success latch and rule."""
    original = env.take_action.__func__
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    guards = []
    successes = []
    for node in ast.walk(function):
        if isinstance(node, ast.If):
            if ast.unparse(node.test) == "self.take_action_cnt == self.step_lim or self.eval_success":
                guards.append(node)
            if ast.unparse(node.test) == "self.check_success()":
                successes.append(node)
    if len(guards) != 1 or len(successes) != 1:
        raise RuntimeError("Unexpected take_action; no diagnostic adaptation applied")
    guards[0].test = ast.parse("self.take_action_cnt == self.step_lim",mode="eval").body
    successes[0].body = ast.parse('''
if not self.eval_success:
    self._native_success_events.append({"command_count": int(self.take_action_cnt), "stage_id": int(self.stage_id)})
self.eval_success = True
''').body
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree,"portable_scoped_continuation","exec"),namespace)
    env._native_success_events = []
    env.take_action = types.MethodType(namespace[function.name],env)
    output=Path(output)
    modified=ast.unparse(tree)
    (output/"scoped_take_action.py").write_text(modified)
    receipt={"original_function_sha256":hashlib.sha256(source.encode()).hexdigest(),"scoped_function_sha256":hashlib.sha256(modified.encode()).hexdigest(),"changes":["remove eval_success from command admission guard; keep command limit","retain native check_success and latched eval_success; log first event instead of returning from physical loop"],"native_success_flag_cleared":False,"official_success_definition_changed":False,"repository_files_changed":False}
    (output/"continuation_adapter.json").write_text(json.dumps(receipt,indent=2))
    return receipt
