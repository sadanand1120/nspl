import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import shutil
import subprocess
import re
import ast
from utilities.std_utils import json_writer, json_reader
from synthesis.dsl import And, Or, Not, Value, ConstComp
from ldips_datagen import LDIPSdatagenSingleEx
import numpy as np


def combine_disjuncts(disjuncts):
    if len(disjuncts) == 1:
        return disjuncts[0]
    elif len(disjuncts) == 2:
        left = disjuncts[0]
        right = disjuncts[1]

        return Or(left, right)
    else:
        left = disjuncts[0]
        right = combine_disjuncts(disjuncts[1:])

        return Or(left, right)


def binop_const_comp_parse(comp):
    if type(comp) is ast.Compare:
        # TODO: Add more ops when needed
        # TODO: Keep an eye out for multiple ops?
        if type(comp.ops[0]) is ast.Gt:
            left = comp.left.id
            right = comp.comparators[0].id
            op = 'Gt'
        elif type(comp.ops[0]) is ast.Eq:
            left = comp.left.id
            right = comp.comparators[0].value
            op = 'Eq'

        return ConstComp(left, right, op)


def binop_values_parse(value):
    if type(value) is ast.Compare:
        # TODO: Add more ops when needed
        # TODO: Keep an eye out for multiple ops?
        if type(value.ops[0]) is ast.Gt:
            left = value.left.id
            right = value.comparators[0].id
            op = 'Gt'
        elif type(value.ops[0]) is ast.Eq:
            left = value.left.id
            if type(value.comparators[0]) is ast.Constant:
                right = value.comparators[0].value
            else:
                right = value.comparators[0].id
            op = 'Eq'
        elif type(value.ops[0]) is ast.Lt:
            left = value.left.id
            if type(value.comparators[0]) is ast.Constant:
                right = value.comparators[0].value
            else:
                right = value.comparators[0].id
            op = 'Lt'

        return Value(left, right, op)


def binop_sub_parse(values, tp):
    if len(values) == 1:
        return binop_values_parse(values[0])
    elif len(values) == 2:
        # TODO: Add more types when needed
        if tp == 'And':
            left = binop_values_parse(values[0])
            right = binop_values_parse(values[1])
            return And(left, right)

    else:
        left = binop_values_parse(values[0])
        right = binop_sub_parse(values[1:], tp)

        # TODO: add more types when needed
        if tp == 'And':
            return And(left, right)


def binop_parse(test):
    # TODO: Add more binop cases
    if type(test.op) is ast.And:
        new_and = And()

        new_and.left = binop_values_parse(test.values[0])

        values = test.values[1:]
        new_and.right = binop_sub_parse(values, 'And')

        return new_and
    elif type(test.op) is ast.Not:
        pred = binop_values_parse(test.operand)
        new_not = Not(pred)

        return new_not
    else:
        print(type(test.op))
        assert False


def get_tests(sketch_ast, path):
    test = None
    for body_idx in path:
        if body_idx == -1:
            break

        if body_idx == -2:
            break

        if type(sketch_ast.body[body_idx]) is ast.If:
            # Add tests
            cur_test = sketch_ast.body[body_idx].test

            # Make sure there are no other corner cases
            if type(cur_test) is ast.BoolOp or type(cur_test) is ast.UnaryOp:
                cur_test = binop_parse(cur_test)

                # Add to Test
                if test == None:
                    test = cur_test
                else:
                    test = And(test, cur_test)
            elif type(cur_test) is ast.Compare:
                # Likely a terrain case
                cur_test = binop_const_comp_parse(cur_test)

                if test == None:
                    test = cur_test
                else:
                    test = And(test, cur_test)
            else:
                assert False

        sketch_ast = sketch_ast.body[body_idx]

    return test


def get_paths(sketch_ast):
    paths = []
    if type(sketch_ast) is ast.Module or type(sketch_ast) is ast.FunctionDef or type(sketch_ast) is ast.If:
        count = 0
        for body in sketch_ast.body:
            sub_paths = get_paths(body)
            if sub_paths != []:
                for sub_path in sub_paths:
                    if type(sub_path) is int:
                        sub_path = [count, sub_path]
                    else:
                        sub_path = [count] + sub_path
                    paths.append(sub_path)

            count += 1

    elif type(sketch_ast) is ast.Return and sketch_ast.value.value == True:
        paths.append(-1)
    elif type(sketch_ast) is ast.Return and sketch_ast.value.value == False:
        paths.append(-2)

    return paths


def make_ldips_ex(ex):
    temp = {}
    for key in ex[1]:
        temp[key] = {
            "dim": [1, 0, 0],
            "type": 'NUM',
            "name": key,
            "value": ex[1][key]
        }

    temp['start'] = {
        "dim": [0, 0, 0],
        "type": "STATE",
        "name": "start",
        "value": "START"
    }

    temp['output'] = {
        "dim": [0, 0, 0],
        "type": "STATE",
        "name": "output",
    }

    # Check if safe
    temp['output']['value'] = ex[0]

    return temp


def parse_sketch(sketch_str):
    sketch_str = sketch_str.replace('None', 'False')
    sketch_str = sketch_str.replace('elif', 'if')
    sketch_ast = ast.parse(sketch_str)

    paths = get_paths(sketch_ast)
    true_paths = []
    false_paths = []
    for path in paths:
        if path[-1] == -1:
            true_paths.append(path)
        else:
            false_paths.append(path)

    pos_disjuncts = []
    for path in true_paths:
        conjuncts = get_tests(sketch_ast, path)
        if conjuncts != None:
            pos_disjuncts.append(conjuncts)

    neg_disjuncts = []
    none_seen = False
    for path in false_paths:
        conjuncts = get_tests(sketch_ast, path)
        if conjuncts != None:
            neg_disjuncts.append(conjuncts)
        else:
            none_seen = True

    # Combine final program
    dsl_sketch_pos = combine_disjuncts(pos_disjuncts)
    if len(neg_disjuncts) > 0:
        dsl_sketch_neg = combine_disjuncts(neg_disjuncts)

        if none_seen:
            not_rest = And(Not(dsl_sketch_neg), Not(dsl_sketch_pos))
            dsl_sketch_neg = Or(dsl_sketch_neg, not_rest)
    else:
        dsl_sketch_neg = Not(dsl_sketch_pos)

    return dsl_sketch_pos.make_ldips_dict(), dsl_sketch_neg.make_ldips_dict(), sketch_str


def extract_values(output):
    # Get Params from both START->SAFE and START->UNSAFE
    start_safe = output[output.find('Adjusted Program:') + len('Adjusted Program:') + 1:output.find('Final Score')]
    temp_out = output[output.find('Final Score') + 1:]
    start_unsafe = temp_out[temp_out.find('Adjusted Program'):temp_out.find('Final Score')]

    # Find Params in Start Safe
    safe_params = []
    idx = start_safe.find('pX')
    temp_sketch = start_safe[idx + 2:]
    while idx > 0:
        param_num = int(temp_sketch[0:re.search(r'(\d)*', temp_sketch).end()])
        val = temp_sketch[temp_sketch.find('[') + 1:temp_sketch.find(']')]
        if (f'pX{param_num:03}', float(val)) not in safe_params:
            safe_params.append((f'pX{param_num:03}', float(val)))
        # Increment
        idx = temp_sketch.find('pX')
        temp_sketch = temp_sketch[idx + 2:]

    # Find Params in Start Unsafe
    unsafe_params = []
    idx = start_unsafe.find('pX')
    temp_sketch = start_unsafe[idx + 2:]
    while idx > 0:
        param_num = int(temp_sketch[0:re.search(r'(\d)*', temp_sketch).end()])
        val = temp_sketch[temp_sketch.find('[') + 1:temp_sketch.find(']')]
        if (f'pX{param_num:03}', float(val)) not in unsafe_params:
            unsafe_params.append((f'pX{param_num:03}', float(val)))
        # Increment
        idx = temp_sketch.find('pX')
        temp_sketch = temp_sketch[idx + 2:]

    # Combine
    params_dict = {}
    safe_dict = dict(safe_params)
    unsafe_dict = dict(unsafe_params)
    all_keys = set(safe_dict.keys()) | set(unsafe_dict.keys())
    for key in all_keys:
        if key in safe_dict and key in unsafe_dict:
            params_dict[key] = (safe_dict[key] + unsafe_dict[key]) / 2
        elif key in safe_dict:
            params_dict[key] = safe_dict[key]
        elif key in unsafe_dict:
            params_dict[key] = unsafe_dict[key]

    # Turn into list
    params = []
    for key in params_dict:
        params.append((key, params_dict[key]))

    return params


def LDIPS_synthesize(examples_data, params_sketch_lsps, pid=0):
    """
    example_data: list of tuples (label, ldips_features)
    params_sketch_lsps: string of sketch with params (pX000, pX001, etc)
    Returns: params tuples list
    """
    ex_list = []
    for ex in examples_data:
        ex_list.append(make_ldips_ex(ex))

    new_sketch_ldips_pos, new_sketch_ldips_neg, new_sketch = parse_sketch(params_sketch_lsps)

    # Write to file
    TEMP_ROOTDIR = os.path.join(nspl_root_dir, f"scripts/synthesis/_ldips_temp_{pid}")
    shutil.rmtree(TEMP_ROOTDIR, ignore_errors=True)
    os.makedirs(os.path.join(TEMP_ROOTDIR, "sketch_dir"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_ROOTDIR, "example_dir"), exist_ok=True)
    json_writer(os.path.join(TEMP_ROOTDIR, "sketch_dir", "START_SAFE.json"), new_sketch_ldips_pos)
    json_writer(os.path.join(TEMP_ROOTDIR, "sketch_dir", "START_UNSAFE.json"), new_sketch_ldips_neg)
    json_writer(os.path.join(TEMP_ROOTDIR, "example_dir", "example.json"), ex_list)

    window_size = 0
    # TODO: Figure out what effect window size has
    # window_size = min((len(ldips_data) - 1), 3)

    args = (os.path.join(nspl_root_dir, "third_party/pips/bin/srtr"),
            "-sketch_dir", os.path.join(TEMP_ROOTDIR, "sketch_dir/"),
            "-ex_file", os.path.join(TEMP_ROOTDIR, "example_dir/example.json"),
            "-window_size", f"{window_size}")

    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = str(popen.stdout.read(), 'UTF-8')
    params = extract_values(output)
    shutil.rmtree(TEMP_ROOTDIR, ignore_errors=True)
    return params


def synthesize_seqn(prev_example_data, new_data_filepath, ldips_synth_program_sketch=None):
    new_data_dict = json_reader(new_data_filepath)
    new_label = new_data_dict['label']
    new_ldips_features = new_data_dict['ldips_features']
    if ldips_synth_program_sketch is None:
        ldips_synth_program_sketch = new_data_dict['ldips_synth_program_sketch']
    prev_example_data.append((new_label, new_ldips_features))
    params = LDIPS_synthesize(prev_example_data, ldips_synth_program_sketch)
    filled_lsps_sketch = LDIPSdatagenSingleEx.fillparams_in_LSPS(ldips_synth_program_sketch, params)
    return prev_example_data, filled_lsps_sketch, params


def read_data(nums_list):
    data_table_dir = os.path.join(nspl_root_dir, "demonstrations/SSR")
    all_json_paths = [os.path.join(data_table_dir, file) for file in sorted(os.listdir(data_table_dir))]
    prev_example_data = []  # list of tuples (label, ldips_features)
    for i in nums_list:
        json_path = all_json_paths[i - 1]
        data = json_reader(json_path)
        prev_example_data.append((data['label'], data['ldips_features']))
    return prev_example_data


if __name__ == "__main__":
    seqn_filled_lfps_sketches = {}
    seqn_lfps_sketches = json_reader(os.path.join(nspl_root_dir, "scripts/llm/seqn_lfps_sketches.json"))
    for i in range(1, 30):
        print(f"Processing {i} of 29")
        nums_list = np.arange(1, i + 1)
        examples_data = read_data(nums_list)
        lfps_sketch = seqn_lfps_sketches[str(i)]
        lsps_sketch = LDIPSdatagenSingleEx.convert_LFPS_to_LSPS(lfps_sketch)
        params = LDIPS_synthesize(examples_data, lsps_sketch)
        filled_lfps_sketch = LDIPSdatagenSingleEx.fillparams_in_LFPS(lfps_sketch, params)
        seqn_filled_lfps_sketches[str(i)] = filled_lfps_sketch
    json_writer(os.path.join(nspl_root_dir, "scripts/llm/seqn_filled_lfps_sketches.json"), seqn_filled_lfps_sketches)
