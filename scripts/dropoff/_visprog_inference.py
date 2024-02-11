import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from third_party.visprog.engine.utils import ProgramGenerator, ProgramInterpreter
from third_party.visprog.prompts.imgedit import PROMPT


def create_prompt(instruction):
    return PROMPT.format(instruction=instruction)


def infer_visprog(pil_img, prompt):
    interpreter = ProgramInterpreter(dataset='imageEdit')
    generator = ProgramGenerator(prompter=create_prompt)
    init_state = dict(
        IMAGE=pil_img.convert('RGB')
    )
    prog, _ = generator.generate(prompt)
    result, prog_state, html_str = interpreter.execute(prog, init_state, inspect=True)
    return result, prog_state, html_str
