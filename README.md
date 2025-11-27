# Loss Landscape Geometry & Optimization Dynamics

This project explores the geometric structure of neural network loss landscapes and how they influence optimization and generalization. We analyze the behavior of SGD, estimate Hessian-based sharpness during training, and visualize 2D loss slices around the converged model.

---

## Project Goal

To develop a rigorous framework connecting:

- Loss surface geometry (via Hessian spectrum)
- Optimization dynamics of SGD
- Model generalization behavior
- Architectural influence (experimented with MLP)

---

##  Experiment Overview

| Component | Description |
|----------|-------------|
| **Dataset** | MNIST (60k training, 10k test) |
| **Model** | 2-layer MLP with 512 hidden units |
| **Optimizer** | SGD (lr=0.1, momentum=0.9) |
| **Epochs** | 5 |
| **Metrics** | Training loss, Sharpness (λ_max), Landscape visualization |

---

###  Model Used

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
