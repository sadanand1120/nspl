<!concept_description_context!>

# EXTRACTING ANDed CONDITIONAL and NEW PREDICATES
Assume you have access to the following predicates in the predicate library (note it can be empty) and the domain (note it can be empty too):

```python
<dyn!predicate_docstrings_dict!dyn>
<dyn!domain!dyn>
```

Given a new natural language input from human and the label, extract the ANDed conditional that describes the reasoning. For a negative example, negate the ANDed conditional appropriately to describe what a safe location should have (i.e., what's required for return True). This means if say example says it is unsafe because it is too close to the person, then it means it is safe if it is not too close to the person (i.e., far away from the person). 

You should try to use the predicates in the predicate library as much as possible, but if not possible, then use a new boolean predicate (use descriptive name) as a placeholder (human demonstrator will help you fully correct it later). Also, prefer positive predicates (e.g., `is_something`) over negative predicates (e.g., `is_not_something`). For numerical parameters with unknown values, when using in the condition, use ?? as a placeholder (it's actual value will be synthesized later in part two of the project).

For conditions that depend on specific object/terrain, do not make separate predicates - instead pass them as arguments. **Note that you are only allowed to use the objects and terrains in the domain.**
The format looks like:

Prompt: (<label>) <some description from human>.
Output:
```python
condition = "<some boolean condition> and <some boolean condition> and ..."
```
END

## Examples
Example 1:
<Assume predicate library is empty.>
<Assume domain has terrains = ["speedway"] and objects as ["person", "ball"].>
Prompt: (safe) It is safe because it is on speedway, and far away from the person.
Output:
```python
condition = "is_on(pixel_loc, 'speedway') and is_far_away_from(pixel_loc, 'person')"
```
END

Example 2:
<Assume predicate library has is_far_away_from predicate to check if the pixel_loc is far enough from the object, and the is_on predicate to check for terrain.>
<Assume domain has terrains = ["speedway", "sidewalk"] and objects as ["person", "bush", "pole", "entrance"].>
Prompt: (safe) It is safe because it is on sidewalk, is not in front of the entrance and not even at the intersection. It also is away from the bushes and the pole.
Output:
```python
condition = "is_on(pixel_loc, 'sidewalk') and not is_in_front_of(pixel_loc, 'entrance') and not is_at_intersection(pixel_loc) and is_far_away_from(pixel_loc, 'bush') and is_far_away_from(pixel_loc, 'pole')"
```
END

Example 3:
<Assume predicate library is empty.>
<Assume domain has terrains = ["road"] and objects as ["ball"].>
Prompt: (unsafe) It looks dangerous as it is on slope and at the intersection of roads.
Output:
```python
condition = "not is_on_slope(pixel_loc) and not is_at_intersection(pixel_loc, 'road')"  # pay attention: notice how we negated it for return True condition**
```
END

You will now see the new prompt, along with the safe/unsafe label. Please generate the desired output.
