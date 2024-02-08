<!concept_description_context!>

# EXTRACTING DOMAIN
Given a new natural language input from human, extract the domain (terrains and objects) from it. What it means is that you need to extract the **terrains** and the **objects** that the human is describing that **play a role in the human's reasoning** for why the location is safe or unsafe. The format looks like:

Prompt: (<label>) <some description from human>.
Output:
```python
objects = ["<some object>", "<some object>", ...]
terrains = ["<some terrain>", ...]
```
END

## Examples
Example 1:
Prompt: (safe) It is safe because it is on speedway, and far away from the person.
Output:
```python
objects = ["person"]
terrains = ["speedway"]
```
END

Example 2:
Prompt: (safe) It is safe because it is on sidewalk, is not in front of the entrance and not even at the intersection. It also is away from the bushes and the pole.
Output:
```python
objects = ["bush", "pole", "entrance"]
terrains = ["sidewalk"]
```
END

Example 3:
Prompt: (unsafe) It looks dangerous as it is on slope and at the intersection of roads.
Output:
```python
objects = []
terrains = ["road"]
```
END

## Strict Guidelines
- You are not allowed to output anything else other than the two python lists. 
- It's okay if you output empty lists if there are no terrains and/or objects.
- Be case-insensitive and singular / plural insensitive (i.e., include only the singular element in the list).
- An object has to be a specific object class (it could be a proper noun too). For instance, "thing" or "item" should NOT count as objects.
- A terrain has to be a specific terrain class of the predefined list above. DO NOT use any other names.

You will now see the new prompt, along with the safe/unsafe label. Please extract the domain from it.
