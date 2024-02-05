"""
Different types of program sketches:
* program_sketch (PS): string for is_safe(pixel_loc) function with the higher level predicates, is_far(pixel_loc, ...) etc
    - filled: string with filled in params with numerical values
    - unfilled: string with params represented as ??
* ldips_func_program_sketch (LFPS): string in the form of ldips state variables, but functional form, terrain(pixel_loc), etc
    - filled: string with filled in params with numerical values
    - unfilled: string with params represented as ??
* ldips_synth_program_sketch (LSPS): string in the form of ldips state variables, with ready-to-synthesize form, terrain == 0, etc
    - filled: string with filled in params with numerical values
    - unfilled: string with params represented as pX###, where ### is a number

Takes in the human NL demonstration and does the following:
- Extracts the label
- Updates the state (json):
    - Updates the domain dict
        => objects list
        => terrains list
    - Updates the program_sketch (PS)
    - Updates the ldips_func_program_sketch (LFPS)
    - Updates the predicate library
        => predicate prototype-docstring dict
        => predicate codestring dict
"""
from simple_colors import red, green
import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import re
from llm._llm import get_llm_response
from utilities.std_utils import reader, json_reader, json_writer, writer, append_writer
import numpy as np


def txt_to_json(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Splitting the content by 'Example' followed by any number
    examples = re.split(r'Example \d+', content)[1:]  # Skip the first split as it will be empty
    # Creating a dictionary without numbered keys
    json_data = {f"{index + 1}": example.strip() for index, example in enumerate(examples)}
    return json_data


class HITLllm:
    def __init__(self, human_nl_demo, state_path, is_first_demo=False):
        self.human_nl_demo = human_nl_demo
        self.state_path = state_path
        self.is_first_demo = is_first_demo
        if self.is_first_demo:
            self.initialize_state()
        self.cur_state = {}
        self.read_cur_state()
        self.PREPROMPTS = {}  # dict of pre-prompts
        self.read_all_preprompts()
        self.label = None

    def initialize_state(self):
        state = {}
        state["domain"] = {
            "objects": [],
            "terrains": [],
        }
        state["program_sketch"] = ""
        state["ldips_func_program_sketch"] = ""
        state["predicate_library"] = {
            "docstrings": {},
            "codestrings": {},
        }
        json_writer(self.state_path, state)

    def read_all_preprompts(self):
        preprompts_dir = os.path.join(nspl_root_dir, "scripts/llm/preprompts")
        for filename in os.listdir(preprompts_dir):
            if filename.endswith(".md"):
                key = filename[:-3]  # Remove the file extension to use as the dictionary key
                file_path = os.path.join(preprompts_dir, filename)
                self.PREPROMPTS[key] = reader(file_path)

    def read_cur_state(self):
        self.cur_state = json_reader(self.state_path)

    def main(self, write_to_json=True):
        """
        Returns:
            writes the new state to json and returns the new_state and the label
        """
        print("Extracting label...")
        self.label = self.extract_label()

        print("Updating domain...")
        self.cur_state["domain"] = self.update_domain()

        print("Extracting conditional...")
        raw_condition = self.extract_raw_conditional()

        print("Extracting new predicates...")
        new_predicates = self.get_new_predicates(raw_condition)

        if len(new_predicates) != 0:
            print("Asking human for predicate explanation...")
            explanation = self.ask_human_for_predicate_explanation(new_predicates)

            print("Generating new predicate docstrings...")
            new_docstrings_dict = self.get_new_proto_docstring_dict(explanation)

            print("Generating new predicate codestrings...")
            new_codestrings_dict = self.get_new_predicate_codestring_dict(new_docstrings_dict)

            print("Updating predicate library...")
            self.cur_state["predicate_library"] = self.update_predicate_library(new_docstrings_dict, new_codestrings_dict)

            print("Updating conditional...")
            updated_condition = self.update_conditional(raw_condition, explanation)
        else:
            print("No new predicates needed. Skipping hitl steps...")
            updated_condition = raw_condition

        print(red(f"Updated conditional: {updated_condition}", 'bold'))

        print("Updating program sketch...")
        progsketch = self.update_program_sketch(updated_condition)

        print("Pruning program sketch...")
        self.cur_state["program_sketch"] = self.prune_program_sketch(progsketch)

        print("Converting program sketch to ldips...")
        self.cur_state["ldips_func_program_sketch"] = self.convert_PS_to_LFPS(self.cur_state["program_sketch"])

        if write_to_json:
            json_writer(self.state_path, self.cur_state)

        return self.cur_state, self.label

    @staticmethod
    def extract_function_names(s):
        pattern = r'\b\w+(?=\()'
        return re.findall(pattern, s)

    @staticmethod
    def extract_function_call_from_name(code, func_name):
        pattern = r'\b' + re.escape(func_name) + r'\([^)]*\)'
        match = re.search(pattern, code)
        return match.group(0) if match else None

    def get_new_predicates(self, raw_condition):
        """
        Uses Regex parsing to get the new predicates from the raw condition string, by comparing against doctsrings in predicate library
        Returns:
            list of new predicates
        """
        current_condition_predicates = self.extract_function_names(raw_condition)
        existing_protos = list(self.cur_state["predicate_library"]["docstrings"].keys())
        existing_predicates = [self.extract_function_names(proto)[0] for proto in existing_protos]
        new_predicates = list(set(current_condition_predicates) - set(existing_predicates))
        new_predicates = [self.extract_function_call_from_name(raw_condition, pred) for pred in new_predicates]
        return new_predicates

    def extract_label(self) -> str:
        """
        Returns:
            a string of the label (safe/unsafe)
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["label"])
        prompt = f"Prompt: {self.human_nl_demo}\nOutput: "
        label = get_llm_response(pre_prompt, prompt)
        label = label.strip()
        label = label.replace("Output: ", "")
        label = label.replace("END", "")
        label = label.strip()
        return label

    def extract_domain(self):
        """
        Returns:
            a dict of the extracted domain
        """
        new_domain = {}
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["domain"])
        prompt = f"Prompt: ({self.label}) {self.human_nl_demo}\nOutput:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        code = match.group(1).strip()
        namespace = {
            "objects": [],
            "terrains": []
        }
        exec(code, namespace)
        new_domain["objects"] = namespace["objects"]
        new_domain["terrains"] = namespace["terrains"]
        return new_domain

    def update_domain(self):
        """
        Returns: 
            updated domain dict
        """
        cur_domain = self.cur_state["domain"]
        cur_domain["objects"] = set(cur_domain["objects"])
        cur_domain["terrains"] = set(cur_domain["terrains"])
        new_domain = self.extract_domain()
        cur_domain["objects"] = cur_domain["objects"].union(new_domain["objects"])
        cur_domain["terrains"] = cur_domain["terrains"].union(new_domain["terrains"])
        cur_domain["objects"] = sorted(list(cur_domain["objects"]))
        cur_domain["terrains"] = sorted(list(cur_domain["terrains"]))
        return cur_domain

    def extract_raw_conditional(self):
        """
        Returns:
            condition string and the new_predicates list
        """
        raw_condition = ""
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["raw_conditional"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           predicate_docstrings_dict=self.cur_state["predicate_library"]["docstrings"],
                                           domain=self.cur_state["domain"])
        prompt = f"Prompt: ({self.label}) {self.human_nl_demo}\nOutput:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        code = match.group(1).strip()
        namespace = {
            "condition": ""
        }
        exec(code, namespace)
        raw_condition = namespace["condition"].strip()
        return raw_condition

    def complete_preprompt(self, value):
        start = value.find("<!")
        while start != -1:
            end = value.find("!>", start + 2)
            if end == -1:  # Malformed placeholder, stop processing
                break
            placeholder = value[start + 2:end]
            replacement = self.PREPROMPTS.get(placeholder, "")
            value = value[:start] + replacement + value[end + 2:]
            start = value.find("<!", end + 2)
        return value

    @staticmethod
    def complete_dynamic(value, keyname_equals=True, **kwargs):
        start = value.find("<dyn!")
        while start != -1:
            end = value.find("!dyn>", start + 5)
            if end == -1:  # Malformed placeholder, stop processing
                break
            placeholder_key = value[start + 5:end]
            replacement_value = kwargs.get(placeholder_key, f"<dyn!{placeholder_key}!dyn>")
            if keyname_equals:
                replacement_str = f"{placeholder_key} = {replacement_value}"
            else:
                replacement_str = f"{replacement_value}"
            value = value[:start] + replacement_str + value[end + 5:]
            # Find the next placeholder only after the replacement we've done
            start = value.find("<dyn!", start + len(replacement_str))
        return value

    def ask_human_for_predicate_explanation(self, new_preds_list):
        """
        Asks human for explanation of the new predicates
        Returns:
            explanation dict
        """
        explanation = {}
        for element in new_preds_list:
            user_input = input(f"Needed {element} predicate. Can you describe?\n")
            explanation[element] = user_input.strip()
        return explanation

    def get_new_proto_docstring_dict(self, explanation):
        """
        Returns:
            new proto-docstring dict
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["proto_gen"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           predicate_docstrings_dict=self.cur_state["predicate_library"]["docstrings"],
                                           explanation=explanation)
        prompt = f"Output:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        code = match.group(1).strip()
        namespace = {
            "new_docstrings_dict": {}
        }
        exec(code, namespace)
        return namespace["new_docstrings_dict"]

    def get_new_predicate_codestring_dict(self, new_docstrings_dict):
        """
        Returns:
            new predicate codestring dict
        """
        new_codestrings_dict = {}
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["code_gen"])
        for proto, docstring in new_docstrings_dict.items():
            pre_prompt_full = self.complete_dynamic(pre_prompt,
                                                    predicate_docstrings_dict=self.cur_state["predicate_library"]["docstrings"],
                                                    new_predicate_prototype=proto,
                                                    new_predicate_docstring=docstring,
                                                    keyname_equals=False)
            prompt = f"Output:\n"
            code = get_llm_response(pre_prompt_full, prompt)
            code = code.strip()
            match = re.search(r'```python(.*?)```', code, re.DOTALL)
            code = match.group(1).strip()
            new_codestrings_dict[proto] = code.strip()
        return new_codestrings_dict

    def update_predicate_library(self, new_docstrings_dict, new_codestrings_dict):
        """
        Returns:
            updated predicate library dict
        """
        cur_predicate_library = self.cur_state["predicate_library"]
        cur_predicate_library["docstrings"].update(new_docstrings_dict)
        cur_predicate_library["codestrings"].update(new_codestrings_dict)
        return cur_predicate_library

    def update_conditional(self, raw_condition, explanation):
        """
        Returns:
            updated conditional string
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["update_conditional"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           human_nl_demo=self.human_nl_demo,
                                           previously_generated_conditional=raw_condition,
                                           previously_proposed_new_predicates_human_responses=explanation,
                                           final_updated_predicate_docstrings_dict=self.cur_state["predicate_library"]["docstrings"])
        prompt = f"Output:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        code = code.replace("Output: ", "")
        code = code.replace("END", "")
        code = code.strip()
        namespace = {
            "updated_condition": ""
        }
        exec(code, namespace)
        return namespace["updated_condition"].strip()

    def update_program_sketch(self, updated_condition):
        """
        Returns:
            updated program sketch string
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["progsketch"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           human_nl_demo=self.human_nl_demo,
                                           final_generated_return_True_conditional=updated_condition,
                                           current_program_sketch=self.cur_state["program_sketch"])
        prompt = f"Output:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        code = match.group(1).strip()
        namespace = {
            "updated_program_sketch": ""
        }
        exec(code, namespace)
        return namespace["updated_program_sketch"].strip()

    def prune_program_sketch(self, progsketch):
        """
        Returns:
            pruned program sketch string
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["critic"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           current_program_sketch=progsketch,
                                           keyname_equals=False)
        prompt = f"Output:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        code = match.group(1).strip()
        return code

    def convert_PS_to_LFPS(self, progsketch):
        """
        Returns:
            ldips-compatible program sketch string
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["ldips"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           current_program_sketch=progsketch,
                                           keyname_equals=False)
        prompt = f"Output:\n"
        code = get_llm_response(pre_prompt, prompt)
        code = code.strip()
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        code = match.group(1).strip()
        return code


if __name__ == "__main__":
    all_yes = False
    NL_FEEDBACK_PATH = os.path.join(nspl_root_dir, "demonstrations/nl_feedback.txt")

    with open(NL_FEEDBACK_PATH, "r") as f:
        nl_feedbacks = f.readlines()

    examples_nums = np.arange(1, 30)  # ordered

    for i in examples_nums:
        print(f"Example {i}")
        nl_feedback = nl_feedbacks[i - 1].strip()
        is_first_demo = True if i == examples_nums[0] else False
        hllm = HITLllm(human_nl_demo=nl_feedback,
                       state_path=os.path.join(nspl_root_dir, "scripts/llm/state.json"),
                       is_first_demo=is_first_demo)
        hllm.main(write_to_json=True)
        print(hllm.cur_state["ldips_func_program_sketch"])
        if not all_yes:
            user_input = input(green("Do you want to continue? (y/n/all)\n", 'bold'))
            if user_input == "n":
                break
            if user_input == "all":
                all_yes = True

        lcl_sketches_path = os.path.join(nspl_root_dir, "scripts/llm/seqn_lfps_sketches.txt")
        if is_first_demo:
            writer(lcl_sketches_path, f"Example {i}\n{hllm.cur_state['ldips_func_program_sketch']}\n\n")
        else:
            append_writer(lcl_sketches_path, f"Example {i}\n{hllm.cur_state['ldips_func_program_sketch']}\n\n")

    json_sketches = txt_to_json(lcl_sketches_path)
    json_writer(lcl_sketches_path.replace(".txt", ".json"), json_sketches)
