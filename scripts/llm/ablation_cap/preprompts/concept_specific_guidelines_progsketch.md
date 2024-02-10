# ROUGH OUTLINE OF THE SKETCH
For the concept we are trying to learn, it makes sense to have separate conditionals for each terrain observed. So we will have a parent conditional `if`/`elif` (note no `else`) statement for each terrain, and then a default `return False`. Within each terrain block, we will have one or more `if`, `elif` statements (note no `else`). Within each of these `if` or `elif` block, we will have a `return True` statement. This is so because in the extreme outermost level, we have a `return False` statement that will be executed if none of the conditions are met. So we need to make sure that we have a `return True` statement in each of the inner `if` or `elif` blocks. Each of the individual inner `if`/`elif` conditions are basically a bunch of boolean conditions `and`'ed.

Summarizing, the sketch should look like:
```python
def is_safe(pixel_loc):
    if <check terrain at the pixel_loc>:
        if <some condition>:
            return True
        elif <some condition>:  # NOT NEEDED mostly, append to the previous if
            return True
        ...
    elif <check another terrain at the pixel_loc>:
        if <some condition>:
            return True
        elif <some condition>:  # NOT NEEDED mostly, append to the previous if
            return True
        ...
    return False
```
where each of the `<some condition>` is a bunch of boolean conditions `and`'ed together.