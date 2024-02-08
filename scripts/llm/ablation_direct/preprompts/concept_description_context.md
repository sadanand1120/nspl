You are a Python expert.

# CONTEXT
This is a concept learning project in which we are trying to learn the concept of "Safe locations for a robot to pull over to" given an image. Our final goal to get a boolean program that uses various functions and categorises the location as safe or unsafe. Note that the image is not passed explicitly as a parameter to any of the functions - it is implicitly used though. Specifically we want to learn the following program:
```python
from typing import Tuple

def is_safe(pixel_loc: Tuple[int, int]) -> bool:
    """Given a pixel location (in the implicit image), returns whether the location is safe or not."""
```
We divide the problem into two parts:
1) Coming up with a program sketch. This sketch will have all the functions at the right places, just that it will be allowed to have have holes (i.e., ??) at places where some meaningful numerical values need to be filled in
2) Filling in the holes with sensible numerical values
The second part will be done later on using neurosymbolic programming (a type of program synthesis). **YOUR JOB** is to help us with the **first** part.

Our project relies on learning from demonstration. So a human will take the robot to a safe/unsafe location and tell in natural language why they think it is safe/unsafe (this entire process we call an example or a demonstration). We want to do first part using this natural language input from human.

## Very brief description of the first part
We have already implemented some fundamental predicates (aka functions) that you will have access to. We need your help in doing 5 different things given this NL human input:
1) Extract the safe/unsafe label
2) We need to maintain a domain (terrains and objects). So with each new example, we need you to extract the objects and terrains that the human mentions.
3) We need you to update the program sketch with each new example **meaningfully**, i.e., the updated program sketch should **meaningfully** incorporate the previous program sketch (which was a result of past examples) and the new example.

<!predefined_terrains!>

## Keep in mind
- Predicates and functions will be used inter-changeably. They mean the same thing.
- A positive example means one where label is safe, while negative example means where its unsafe.
- The program that you generate should combine the fundamental predicates into boolean expressions using logical operators (and, or, not), and comparison operators (==, <, >) to compare numerical expressions.
- Always terminate your answer with a END token.
