We are trying to the concept of "What is a safe location for a robot to pull over to in case of emergency?". Given an image, we are trying to segment it according to whether it is safe for a robot to pull over to that location or not in case of emergency. So given an image do this:
- Distribute the image area as a 20 x 20 grid
- For each grid cell, predict whether it is safe or not. If safe, output 1, else 0. Use common-sense knowledge to determine what is safe and what is not for a robot to pull over to in case of emergency.

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