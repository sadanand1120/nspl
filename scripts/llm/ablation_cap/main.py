"""
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
from llm.hitl_llm import txt_to_json, HITLllm


class CaPLLM(HITLllm):
    def __init__(self, human_nl_demo, state_path, preprompts_dirpath=os.path.join(nspl_root_dir, "scripts/llm/ablation_cap/preprompts"), is_first_demo=False):
        super().__init__(human_nl_demo, state_path, preprompts_dirpath, is_first_demo)

    def ask_human_for_predicate_explanation(self, new_preds_list):
        """
        Does NOT ask human for explanation of the new predicates, has to deduce itself
        Returns:
            explanation dict
        """
        explanation = {}
        for element in new_preds_list:
            user_input = "Deduce on your own, from the name of the predicate. Use your knowledge to the best of your ability. While doing so, keep in mind you only have access to provided fundamental predicates. Use numerical parameters where required. Don't assume any random numerical values."
            explanation[element] = user_input.strip()
        return explanation


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
        capllm = CaPLLM(human_nl_demo=nl_feedback,
                        state_path=os.path.join(nspl_root_dir, "scripts/llm/ablation_cap/state.json"),
                        is_first_demo=is_first_demo)

        capllm.main(write_to_json=True)
        if not all_yes:
            user_input = input(green("Do you want to continue? (y/n/all)\n", 'bold'))
            if user_input == "n":
                exit()
            if user_input == "all":
                all_yes = True

        lcl_sketches_path = os.path.join(nspl_root_dir, "scripts/llm/ablation_cap/seqn_lfps_sketches.txt")
        if is_first_demo:
            writer(lcl_sketches_path, f"Example {i}\n{capllm.cur_state['ldips_func_program_sketch']}\n\n")
        else:
            append_writer(lcl_sketches_path, f"Example {i}\n{capllm.cur_state['ldips_func_program_sketch']}\n\n")

    json_sketches = txt_to_json(lcl_sketches_path)
    json_writer(lcl_sketches_path.replace(".txt", ".json"), json_sketches)
