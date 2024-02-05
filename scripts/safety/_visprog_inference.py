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
    # with open("/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/scripts/safety/example.html", "w") as file:
    #     file.write(html_str)
    return result, prog_state, html_str


if __name__ == "__main__":
    img_path = '/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/third_party/visprog/assets/ut.png'
    # prompt = "Replace safe location for a robot to pull over to with snow."
    prompt = "Safe location = sidewalk or concrete, and far away from objects. Replace safe location with snow."
    pil_img = Image.open(img_path)
    result, prog_state, html_str = infer_visprog(pil_img, prompt)
    # plt.imshow(result)
    # plt.show()
    unified_mask = np.any([mask_dict['mask'] != 0 for mask_dict in prog_state['OBJ1']], axis=0).astype(np.uint8)
    plt.imshow(unified_mask)
    plt.show()
