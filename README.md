---
license: apache-2.0
---

[(https://colab.research.google.com/drive/1p-czYknUS1gxF-63CIs1PeKQZpcbtO0s)](https://colab.research.google.com/drive/1p-czYknUS1gxF-63CIs1PeKQZpcbtO0s?usp=sharing)

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

This journey from a boring, static cube to a vibrant, dynamic Rubik's Cube makes the world of embeddings much more fascinating and (maybe more) effective. Experiment test .. test experiment..  Don't stop experimenting because some person said its not worth your time.. Don't copy and paste a basic transformer.. thats boring and for old people.


class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, n_state, n_head, num_rotations, base=10000, checkpointing=False):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        self.num_rotations = num_rotations
        self.base = base
        self.checkpointing = checkpointing
        
        self.thetas = nn.Parameter(torch.zeros(num_rotations))
        self.rotation_pairs = nn.Parameter(data=torch.rand(num_rotations, 2) * self.h_dim)
        self.theta_scale = nn.Parameter(data=torch.ones(1))  
        self.rotation_matrix = nn.Parameter(data=torch.eye(n=self.h_dim))
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
        
        # Adding scaling factor for num_rotations
        self.num_rotations_scale = nn.Parameter(data=torch.ones(1))
    
    def givens_rotation_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state, device=theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G
    
    def update_base(self, new_base):
        self.base = float(new_base)
        self.base = new_base
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
    
    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.rotation_matrix)
        nn.init.zeros_(tensor=self.thetas)
    
    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
        
        if x.dim() == 3:
            batch_size, seq_len, n_state = x.size()
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        else:
            batch_size, seq_len, n_head, h_dim = x.size()
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(f"Expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
        
        x = x.reshape(-1, self.h_dim)
        
        # Adjust num_rotations based on scaling factor
        adjusted_num_rotations = int(self.num_rotations * self.num_rotations_scale.item())
        
        for k in range(adjusted_num_rotations):
            i, j = self.rotation_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale  
            G = self.givens_rotation_matrix(n_state=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=G)
        
        x = torch.matmul(input=x, other=self.rotation_matrix)
        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        
        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device), self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.n_state)
        return x

