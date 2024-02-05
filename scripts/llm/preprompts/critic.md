<!concept_description_context!>

# CRITIC CHECK
The entire program sketch has been synthesized. Now you are a critic. You need to check for any redundant (i.e., repeated same condition ANDed together, etc) or unreachable/dead code (i.e., code that will never be executed). If you find any such code:
- in case of redundancy, remove the redundant part only
- in case of unreachable code, remove the **ENTIRE** block that's unreachable, and NOT only the condition that makes it unreachable
Note, we always have a default return value of `False` at the end of the program sketch.

Examples:
Example 1: Redundant code (prune the repeated ANDed condition)
Input: ... is_far_away_from(pixel_loc, 'person', ??) and is_far_away_from(pixel_loc, 'pole', ??) and is_far_away_from(pixel_loc, 'person', ??) ...
Pruned: ... is_far_away_from(pixel_loc, 'person', ??) and is_far_away_from(pixel_loc, 'pole', ??) ...

Example 2: Unreachable code (prune the entire block)
Input:
```python
def is_safe(pixel_loc):
    
    ...

    elif is_on(pixel_loc, 'sidewalk'):
        if not is_on(pixel_loc, 'sidewalk') and ...:
            return True  # cannot be reached ever
    
    ...

    return False
```
Pruned:
```python
def is_safe(pixel_loc):
    
    ...


    ...

    return False
```

Example 3: Unnecessary code (prune the entire block)
Input:
```python
def is_safe(pixel_loc):
    
    ...

    elif ...:
        if ...:
            return False  # unnecessary block since we have a default return value of False at the end
    
    ...

    return False
```
Pruned:
```python
def is_safe(pixel_loc):
    
    ...


    ...

    return False
```

Example 4: Redundant code
Input:
```python
def is_safe(pixel_loc):
    
    ...

    elif is_on(pixel_loc, 'sidewalk'):
        if not is_on(pixel_loc, 'grass') and ...:  # if terrain is sidewalk, it cannot be grass
            return True
    
    ...

    return False
```
Pruned:
```python
def is_safe(pixel_loc):
    
    ...

    elif is_on(pixel_loc, 'sidewalk'):
        if ...:
            return True
    
    ...

    return False
```

You need to finally output the pruned program sketch. If everything is fine, you can output the same program sketch.

The format of the output looks like:
Output:
```python
<pruned program sketch>
```
END

Given the following program sketch:
```python
<dyn!current_program_sketch!dyn>
```
Please convert it to the pruned form.
