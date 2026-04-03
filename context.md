# Context and Objective
Act as a Senior Deep Learning Researcher and RL Engineer. I am conducting a 1-week Proof of Concept (PoC) experiment to find mechanistic evidence of emergent "Theory of Mind" (ToM) in model-free Reinforcement Learning, inspired by Bush et al. (2025). 

Instead of a complex game, we are building a minimal "Forked Corridor Coordination" environment. The core design principle is **partial observability over intent, not state.** Agents share the same physical world but have hidden private goals. They must infer their partner's goal from early movement to avoid costly collisions.

We will execute this in three phases (V1-V3). Please provide the code modularly so we can test V1 before moving to V2.

---

## Phase V1 (Days 1-2): The Environment & Baseline PPO
Write `env.py` and a basic `train_v1.py`. 

### 1. The "Forked Corridor" Environment (PettingZoo Parallel API)
Build a custom, vectorized PettingZoo `ParallelEnv` using PyTorch/NumPy.
* **Grid Layout (7x7):**
  #######
  #  L R#
  #  | |#
  #  .  #
  #  S  #
  #  .  #
  #######
  * `S`: Spawn zone (bottom center).
  * `.`: Shared open area/corridor.
  * `|`: Narrow 1-tile bottlenecks.
  * `L` & `R`: The two goal zones.
* **The Hidden Variable (Latent Intent):** At `env.reset()`, randomly assign each agent a private target goal (L or R). **Crucial:** Agent A cannot see Agent B's assigned goal, and vice versa.
* **Observations:** Return a tensor `(7, 7, C)` containing spatial channels (Walls, Self, Partner). Include the agent's *own* assigned goal as a non-spatial vector concatenated to the flattened CNN output, but explicitly exclude the partner's goal.
* **Actions:** Up, Down, Left, Right, Stay.
* **Rewards (Asymmetric Payoff + Collision Penalty):**
  * +10 for reaching their assigned private goal.
  * -5 (and episode termination) if both agents enter the same narrow `|` corridor simultaneously (collision).
  * -0.1 step penalty.
* **Randomization:** Randomize the exact spawn coordinates within the `S` zone slightly to prevent degenerate memorization. The shared `.` space before the fork must be long enough that early moves are slightly ambiguous, forcing agents to wait and observe before committing.

### 2. Baseline Training (Standard PPO)
* Write a CleanRL-style MAPPO training loop.
* For V1, use standard CNN -> MLP Actor and Critic networks. 
* Prove the environment is solvable. The agents should learn to observe each other in the shared space, deduce the other's path, and coordinate to get their +10 rewards without colliding.

---

## Phase V2 (Days 3-5): The DRC Integration
Once V1 is confirmed working, write `models_drc.py` and `train_v2.py`.

* **Upgrade the Actor:** Replace the standard CNN with the Deep Repeated ConvLSTM (DRC) architecture from Bush et al. (2025).
* **Hyperparameters:** $D=2$ layers, $N=2$ internal ticks, $G_d=32$ channels.
* **Execution:** Train the DRC MAPPO stack on the Forked Corridor until it achieves a high success rate (avoiding bottlenecks, reaching goals). Save the model weights.

---

## Phase V3 (Days 5-7): Probing and Intervention
Write `interpretability.py` to test the trained V2 model.

### 1. The Global Goal Probe
* Generate 1,000 evaluation rollouts using the frozen DRC actors. Record Agent A's internal hidden states at each step.
* **Target Concept:** `partner_goal` $\in \{L, R\}$ (Binary).
* **The Test:** Train a linear classifier (logistic regression) on Agent A's final-tick cell state at timestep $t=2$ or $t=3$ (in the shared corridor, before the fork).
* Evaluate using Macro F1. If the probe accurately predicts Agent B's hidden goal, we have found an internal ToM representation.

### 2. Causal Intervention (The "Inception" Test)
* Identify the learned concept vector $W_L$ (the vector corresponding to the partner going LEFT).
* Pick a rollout where Agent B is clearly assigned RIGHT. 
* At timestep $t=3$, manually inject the $W_L$ vector into Agent A's hidden state ($h_{new} = h_{old} + W_L$).
* **Measurement:** Check if Agent A's behavioral distribution shifts (e.g., does it suddenly avoid the Left corridor and route Right, hallucinating that its partner is going Left?).

---

## Performance Optimizations
We are running on an L40S GPU. Please implement:
1. Vectorized environments (e.g., 512+ parallel envs).
2. `torch.compile(mode="reduce-overhead")` for the Actor/Critic forward passes.
3. `torch.autocast` (bfloat16) and `torch.backends.cuda.matmul.allow_tf32 = True` to maximize tensor core utilization.


### V2 Architecture Addendum: Exact DRC(3,3) Specifications

Please build the DRC module matching the precise specifications of the DRC(3,3) agent from Guez et al. (2019) and Bush et al. (2025). Do not import an external repository; write this from scratch in PyTorch. 

We will adjust the spatial dimensions to fit our 7x7 grid, but all internal routing must perfectly match the paper's equations.

#### 1. Global Architecture Hyperparameters
* **ConvLSTM Depth ($D$):** 3 stacked layers.
* **Internal Ticks ($N$):** 3 ticks per environment step.
* **Channel Depth ($G_d$):** 32 channels for the encoder and all ConvLSTM hidden/cell states.
* **Kernel Size:** 3x3 for all convolutions, with `padding=1` to strictly preserve the 7x7 spatial dimensions at every layer.

#### 2. Network Components & The "Tick" Routing
**A. The Encoder:**
* A single `nn.Conv2d` taking the environment observation. Output: $i_t \in \mathbb{R}^{32 \times 7 \times 7}$.

**B. The Deep Repeated ConvLSTM (DRC) Core:**
* Create a `ConvLSTMCell` using `nn.Conv2d`. 
* **The "Pool-and-Inject" Mechanism (CRITICAL):** Each ConvLSTM cell must maintain a custom projection layer. Before tick $n$, take the cell's output from the prior tick $h_{t, n-1}^d$. 
  1. Apply Global Mean Pooling and Global Max Pooling spatially.
  2. Concatenate them into a flat vector of size $2G_d = 64$.
  3. Pass through a Linear layer (Affine transformation) to size $H \times W \times G_d$ (which is $7 \times 7 \times 32 = 1568$).
  4. Reshape back to $(32, 7, 7)$ and inject/add it to the cell's computation for the current tick.
* **The "Tick" Loop ($n=1$ to $3$):**
  * **Bottom-Up Skips:** The encoder output $i_t$ is passed as an additional input to *every* ConvLSTM layer at *every* tick.
  * **Top-Down Skips:** The output of Layer 3 from the *previous* tick ($h_{t, n-1}^3$) is passed as an additional input to Layer 1 on the *current* tick.

**C. The Output Heads:**
* After 3 ticks, take the final hidden state of Layer 3 ($h_{t, 3}^3$), concatenate it with the original encoder output $i_t$, and apply an Affine transformation + ReLU to get a flattened activation vector $o_t$.
* **Actor:** Linear layer on $o_t$ to output 5 action logits.
* **Critic:** Linear layer on $o_t$ to output 1 state-value.

#### 3. PPO / Training Hyperparameters (Adapted from Appendix E.4)
The paper used IMPALA, but for our CleanRL MAPPO setup, use these aligned hyperparameters:
* **Optimizer:** Adam.
* **Learning Rate:** Linear decay from $4e^{-4}$ to $0$.
* **Batch Size:** Ensure micro-batches are appropriately sized for your vectorized environments (e.g., if using 512 envs, unroll length of 20).
* **Discount ($\gamma$):** 0.97.
* **Entropy Penalty:** 0.01 (1e-2).
* **L2 Regularization:** Apply weight decay of $1e^{-5}$ to the policy and value heads.