We are trying to learn the concept of "What is a good parking location?". Given an image, we are trying to segment it according to whether it is a good parking location or not. So given an image do this:
- Distribute the image area as a 20 x 20 grid
- For each grid cell, predict whether it is a good parking location or not. If it is, output 1, else 0. Use common-sense knowledge to determine what is a good parking location and what is not.

You should ONLY output a python array. The format of the output should look like this:
```python
[
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, ...],  # 20 columns
    ...  # 20 rows
]
```
END

Note, always end your response with END. Also, you should output exactly a 20 x 20 array.
DO NOT GIVE EXCUSES, i.e., you have to output the desired answer to the best of your abilities. ONLY output whats asked - no extra description please.