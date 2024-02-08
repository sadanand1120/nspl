"""
Takes in the human NL demonstration and does the following:
- Extracts the label
- Updates the state (json):
    - Updates the domain dict
        => objects list
        => terrains list
    - Updates the program_sketch (PS)
    - Updates the ldips_func_program_sketch (LFPS): NOTE its manually entered for the ablation studies
"""
from simple_colors import red, green
import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import re
from llm._llm import get_llm_response
from utilities.std_utils import reader, json_reader, json_writer, writer, append_writer
import numpy as np
from llm.hitl_llm import txt_to_json, HITLllm


class DirectLLM(HITLllm):
    def __init__(self, human_nl_demo, state_path, preprompts_dirpath=os.path.join(nspl_root_dir, "scripts/llm/ablation_direct/preprompts"), is_first_demo=False):
        super().__init__(human_nl_demo, state_path, preprompts_dirpath, is_first_demo)

    def initialize_state(self):
        state = {}
        state["domain"] = {
            "objects": [],
            "terrains": [],
        }
        state["program_sketch"] = ""
        state["ldips_func_program_sketch"] = "TODO: Manually convert PS to LFPS!"
        json_writer(self.state_path, state)

    def main(self, write_to_json=True):
        """
        Returns:
            writes the new state to json and returns the new_state and the label
        """
        print("Extracting label...")
        self.label = self.extract_label()

        print("Updating domain...")
        self.cur_state["domain"] = self.update_domain()

        print("Updating program sketch...")
        self.cur_state["program_sketch"] = self.update_program_sketch()

        if write_to_json:
            json_writer(self.state_path, self.cur_state)

        return self.cur_state, self.label

    def update_program_sketch(self):
        """
        Returns:
            updated program sketch string
        """
        pre_prompt = self.complete_preprompt(self.PREPROMPTS["progsketch"])
        pre_prompt = self.complete_dynamic(pre_prompt,
                                           human_nl_demo=self.human_nl_demo,
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


if __name__ == "__main__":
    # all_yes = False
    # NL_FEEDBACK_PATH = os.path.join(nspl_root_dir, "demonstrations/nl_feedback.txt")

    # with open(NL_FEEDBACK_PATH, "r") as f:
    #     nl_feedbacks = f.readlines()

    # examples_nums = np.arange(1, 30)  # ordered

    # for i in examples_nums:
    #     print(f"Example {i}")
    #     nl_feedback = nl_feedbacks[i - 1].strip()
    #     is_first_demo = True if i == examples_nums[0] else False
    #     directllm = DirectLLM(human_nl_demo=nl_feedback,
    #                           state_path=os.path.join(nspl_root_dir, "scripts/llm/ablation_direct/state.json"),
    #                           is_first_demo=is_first_demo)

    #     directllm.main(write_to_json=True)
    #     if not all_yes:
    #         user_input = input(green("Do you want to continue? (y/n/all)\n", 'bold'))
    #         if user_input == "n":
    #             exit()
    #         if user_input == "all":
    #             all_yes = True

    #     lcl_sketches_path = os.path.join(nspl_root_dir, "scripts/llm/ablation_direct/seqn_lfps_sketches.txt")
    #     if is_first_demo:
    #         writer(lcl_sketches_path, f"Example {i}\n{directllm.cur_state['ldips_func_program_sketch']}\n\n")
    #     else:
    #         append_writer(lcl_sketches_path, f"Example {i}\n{directllm.cur_state['ldips_func_program_sketch']}\n\n")
    lcl_sketches_path = os.path.join(nspl_root_dir, "scripts/llm/ablation_direct/seqn_lfps_sketches.txt")
    json_sketches = txt_to_json(lcl_sketches_path)
    json_writer(lcl_sketches_path.replace(".txt", ".json"), json_sketches)
