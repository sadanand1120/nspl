# Final desired form of the predicates
Each predicate should be a boolean expression composed of a combination of the following:
- a state variable, that depends on just the pixel location
- a binary comparison operator
- a parameter (known constant or unknown)
Thus, the final desired form of each of the predicates is:
```python
<state_var> <binary_comparison_operator> <parameter>
```

## State variables
You only have access to following state variables (the names have important meaning):
- distance_to_<object_class>(pixel_loc): Here, you'll have to the <object_class> hole according to the predicate context. This will typically be compared to a unknown parameter.
- terrain(pixel_loc): This should be compared to a known parameter, i.e., specifically one of the terrain integer labels.
- frontal_distance_<object_class>(pixel_loc): Here, you'll have to the <object_class> hole according to the predicate context. This will typically be compared to a unknown parameter.
- in_the_way(pixel_loc): This is a boolean. So, should be compared to either 0 (False) or 1 (True).
- slope(pixel_loc): This will typically be compared to a unknown parameter.

If you have forgotten, remember the following:
<!predefined_terrains!>

## Binary comparison operators
You only have access to `==`, `<`, `>` binary comparison operators.

## Parameters
For known, use the actual value. For unknown, use "??" as a placeholder/hole that will be filled later during synthesis (part 2 of the project).
