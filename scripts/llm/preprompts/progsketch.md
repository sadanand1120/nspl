<!concept_description_context!>

<!concept_specific_guidelines_progsketch!>

# UPDATING PROGRAM SKETCH for is_safe
Given the return-True ANDed conditional that you have deduced from the given human NL input, you need to update the current program sketch in the most meaninful way. The format of the output looks like:

Output:
```python
updated_program_sketch = "<updated program sketch>"
```
END

## Important Rules
- You should strictly adhere to the program sketch guidelines mentioned above.
- Note, for the concept we are trying to learn, for each terrain block, it mostly makes sense to have a single `if` inside it. You should always try and append the ANDed conditional to the inner if as much as possible, making sure you DO NOT repeat any condition in the same ANDed group.
- If there's a ANDed conditional without any terrain specification, then add it to all the terrain blocks appropriately.
- DO NOT have any redundant code.
- DO NOT have any redundant conditions.
- DO NOT have any redundant `return False` statement. There should be one and only one `return False` statement, that is at the outermost level at the very end.

This is the final step. You have done all the previous steps and now you have the following information:

```python
# Human NL demonstration input
<dyn!human_nl_demo!dyn>

# Final generated (return-True) ANDed conditional deduced from the human NL input
<dyn!final_generated_return_True_conditional!dyn>

# Current Program Sketch
<dyn!current_program_sketch!dyn>
```

Please generate the desired output.
