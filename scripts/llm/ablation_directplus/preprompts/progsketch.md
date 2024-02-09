<!concept_description_context!>

# UPDATING PROGRAM SKETCH for is_safe
Given the new human NL input, you need to update the current program sketch in the most meaningful way. The format of the output looks like:

Output:
```python
updated_program_sketch = "<updated program sketch>"
```
END

You have access to the following functions (ONLY use these in your sketch):
<!fundamental_preds!>

## Important Rules
- If you need some numerical parameter somewhere, use ?? as a placeholder.
- YOU ARE ONLY ALLOWED to use the functions defined in the dsl above.
- The sketch you generate should have a default return False at the very end. Before that you can have any number of `if` statements.
- DO NOT have any redundant code.
- DO NOT have any redundant conditions.

This is the final step. You have done all the previous steps and now you have the following information:

```python
# Human NL demonstration input
<dyn!human_nl_demo!dyn>

# Current Program Sketch
<dyn!current_program_sketch!dyn>
```

Please generate the desired output.
