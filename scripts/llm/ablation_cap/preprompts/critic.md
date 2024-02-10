<!concept_description_context!>

# CRITIC CHECK
The entire program sketch has been synthesized. Now you are a critic. You need to check for any redundant (i.e., repeated same condition ANDed together, etc) or unreachable code (i.e., code that will never be executed). If you find any such code:
- in case of redundancy, remove the redundant part
- in case of unreachable code, remove the **ENTIRE** block that's unreachable, and NOT the condition alone that makes it unreachable

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
