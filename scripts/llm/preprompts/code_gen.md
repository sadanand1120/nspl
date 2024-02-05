<!concept_description_context!>

<!fundamental_preds!>

# GENERATE CODE for NEW PREDICATES
Assume you have access to all the fundamental predicates as well as the predicate library:

```python
predicate_docstrings_dict = <dyn!predicate_docstrings_dict!dyn>
```

and that you can use ONLY these available functions. Given a new predicate prototype and associated docstring in the following format:

Prompt:
```python
def <new predicate full prototype>:
    """<new predicate docstring>"""
```

you need to generate the code for the new predicate. Output the entire function definition, i.e., along with function prototype, docstring, and the function body, in the following format:

Output:
```python
def <new predicate full prototype>:
    """<new predicate docstring>"""
    <new predicate function body>
```
END

Now, you are given the following new predicate and doctsring. Please generate the desired output:

Prompt:
```python
def <dyn!new_predicate_prototype!dyn>:
    """<dyn!new_predicate_docstring!dyn>"""
```
