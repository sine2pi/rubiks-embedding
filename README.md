---
license: apache-2.0
---
[
(https://colab.research.google.com/drive/1p-czYknUS1gxF-63CIs1PeKQZpcbtO0s)

Drop-in enhanced givens rotary block --  Its like a rubiks cube of embbedings :)

Imagine two intrepid tensors, embarking on a thrilling quest through this ever-changing Rubik's Cube. They navigate through a labyrinth of rotations, guided by a dynamic 
rulebook that adjusts the number of twists and turns along the way. It's like a choose-your-own-adventure through a kaleidoscope of possibilities!

### **Regular Rotary Embedding (3D Cube in Space)**

Picture a plain old cube just hanging out in space—static, predictable, and, let's be honest, a bit boring. This is your regular rotary embedding. It has a few key steps:

1. **Initialization**:
   - Parameters are set up to represent angular frequencies and phases for each dimension of our data cube.

2. **Sinusoidal Embedding**:
   - Each face of the cube gets assigned a sinusoidal pattern, encoding positional information to help the model understand the order and relationship between tokens.

3. **Rotation**:
   - We rotate the cube using defined angles (theta), adjusting the positions of points (tokens) in this multidimensional space.

4. **Output**:
   - After rotation, the new positions of points represent transformed embeddings, carrying both their original information and positional context.

### **Combined Rotary Embedding (Dynamic Rubik's Cube)**

Now, let's turn up the excitement and transform our plain cube into a Rubik's Cube, constantly twisting and turning in response to dynamic rules—vibrant and full of surprises!

1. **Initialization**:
   - More sophisticated than our basic 3D cube, we have parameters like `thetas`, `rotation_pairs`, `theta_scale`, and `num_rotations_scale`, serving as the mechanisms controlling our rotations.

2. **Givens Rotation Matrix**:
   - This matrix defines individual rotations within our Rubik's Cube, allowing us to rotate specific pairs of dimensions (faces).

3. **Dynamic Adjustment**:
   - The `num_rotations_scale` parameter dynamically adjusts how many rotations we apply, changing the rules for how many moves we can make to solve our Rubik's Cube.

4. **Rotation Loop**:
   - For each adjusted rotation, we apply a specific Givens rotation matrix, transforming our position within the cube.

5. **Final Rotation**:
   - After all Givens rotations, a final rotation with the `rotation_matrix` ensures our data is in its final transformed state.

6. **Sinusoidal Embedding**:
   - Similar to our basic 3D cube, we apply sinusoidal embeddings to capture positional information, helping the model retain context about the order and relationship between tokens.

7. **Output**:
   - The final output is a dynamically rotated and transformed embedding, carrying both the original token information and enhanced positional context, ready for the next steps in the model's processing pipeline.

---

So, the **Combined Rotary Embedding** is like a Rubik's Cube with dynamic moves, allowing for more complex and adaptable rotations, which can potentially improve the model's performance by better capturing positional relationships within the data.

This journey from a boring, static cube to a vibrant, dynamic Rubik's Cube makes the world of embeddings much more fascinating and (maybe more) effective. Experiment test .. test experiment..
