# Reason Theory Notes

This note formalizes the current `reason` scoring logic for
`partially_occluded` and `fully_occluded` targets, and proposes a
volume-aware probability model that can be turned into a ranking option later.

The user's wording "partial-occupied / fully-occupied" is treated here as the
code's `partially_occluded / fully_occluded` branches.

## Sources Read

- Tang et al. 2025, *AffordGrasp: In-Context Affordance Reasoning for
  Open-Vocabulary Task-Oriented Grasping in Clutter*.
  Key idea: VLM reasoning decomposes language and image into task, object,
  graspable part, and affordance: `T, O, p*, a* = GPT-4o(L, I)`.
- Qian et al. 2026, *ThinkGrasp: A Vision-Language System for Strategic Part
  Grasping in Clutter*.
  Key idea: high-level target selection is an argmax over language, scene
  context, and candidate object, while considering task relevance, ease of
  grasping, and obstruction.
- Jiao et al. 2025, *Free-form Language-based Robotic Reasoning and Grasping*.
  Key idea: occlusion graphs provide the action sequence. Edges below an
  occlusion-area threshold are pruned, then traversal from target to a
  top-level/leaf obstructor gives the first object to grasp.
- Bejjani et al. 2021, *Occlusion-Aware Search for Object Retrieval in
  Clutter*.
  Key idea: hidden-target retrieval under occlusion is naturally a belief-space
  problem; actions update a probability distribution over possible states.
- Lei et al. 2026, *ActiveGrasp: Information-Guided Active Grasping with
  Calibrated Energy-based Model*.
  Key idea: grasp-oriented information gain can be defined as entropy reduction
  of a task-specific distribution, rather than as a generic visibility score.
- Breyer et al. 2021, *Volumetric Grasping Network*.
  Key idea: a TSDF volume can carry voxel-level grasp quality, orientation, and
  width, which motivates replacing the current 2.5D volume proxy with a real
  volumetric representation later.

## Current Variables

Let:

- `G = (V, E)` be the directed occlusion graph.
- Edge `u -> v` means object `u` occludes object `v`.
- `r_uv in [0, 1]` is the edge contact ratio stored as `ratio`.
- `t` is the target object id.
- `C` is the currently graspable top-layer candidate set.
- `s_i` is the VLM semantic score/probability for candidate `i`.
- `g_i` is the geometry score for candidate `i`.
- `q_i` is the graspability score for candidate `i`, usually
  `max_part q_{i,part}` when SAM2/VLM part scores are available.
- `P(i)` is the normalized belief used by the ranking module.
- `H(P) = - sum_i P(i) log2 P(i)` is Shannon entropy in bits.

The shared fusion pattern is a product-of-experts posterior:

```Markdown
raw_i = s_i^beta * g_i^gamma
P(i) = raw_i / sum_j raw_j
```

The current implementation uses `beta = gamma = 1`.

This is reasonable because the VLM and geometry terms behave like two different
experts: the VLM scores semantic/task relevance, while the graph/depth/mask term
scores physical plausibility. Normalizing their product converts independent
scores into a decision belief over candidates.

## Partially Occluded Target

The target is visible in the graph but has ancestors. The policy may only remove
top-layer ancestors now:

```text
A_t = ancestors_G(t)
C_t = { i in A_t : in_degree_G(i) = 0 }
```

### Semantic Prior

`reason/partially_visible/prior.py` asks the VLM to score all ancestors of the
target. These scores are independent, not mutually exclusive:

```text
s_i ~= p(candidate i is important in the occlusion chain | L, I, G)
```

Only top-layer scores are used for the immediate action, but lower-layer scores
are cached for counterfactual entropy after removing one object.

### Geometric Prior

The current geometry score sums path products:

```text
g_i = sum_{path pi: i -> t} product_{(u,v) in pi} r_uv
```

Interpretation: if each edge ratio is treated as the chance or strength that
one object materially blocks the next object, a path product is the obstruction
strength along one chain. Summing over paths approximates the total graph
support that `i` blocks `t`.

A bounded variant is mathematically cleaner:

```text
path_strength(pi) = product_{(u,v) in pi} r_uv
g_i = 1 - product_{pi: i -> t} (1 - path_strength(pi))
```

The current sum is a first-order approximation of this union probability. Since
the code normalizes after fusion, the current version is usable, but the bounded
version avoids scores greater than 1 and is easier to defend in a paper.

### Belief and Entropy

For the current top-layer candidates:

```text
P_0(i) = normalize_i(s_i * g_i), i in C_t
H_0 = H(P_0)
```

The entropy is uncertainty over the next best top-layer occlusion-chain
bottleneck. Lower entropy means the policy has a clearer candidate to remove.

### Counterfactual Structural Information Gain

If action `a` removes candidate `a`, construct a residual graph:

```text
G_-a = G without node a
C_-a = top-level ancestors of t in G_-a
```

If no candidate remains, the target is directly graspable:

```text
H_-a = 0
```

Otherwise rebuild a posterior on the new candidate set, reusing cached VLM
ancestor scores:

```text
P_-a(i) = normalize_i(s_i * g_i(G_-a)), i in C_-a
H_-a = H(P_-a)
IG_partial(a) = H_0 - H_-a
```

This is valid as a *structural entropy reduction* objective. It is not a strict
mutual information between the same random variable before and after action,
because the candidate set can change after removing an object. The defensible
interpretation is:

```text
IG_partial(a) = uncertainty of current subproblem
                - uncertainty of residual subproblem after removing a
```

The existing legacy utility is:

```text
U_legacy(a) = P_0(a) * IG_partial(a) - alpha * cost(a)
```

The current experiment variants are:

```text
U_ig(a) = IG_partial(a)
U_ig_graspability(a) = IG_partial(a) * q_a
```

Recommended paper-ready form:

```text
IGn(a) = max(0, IG_partial(a)) / log2(max(2, |C_t|))
U_partial(a) = q_a * P_0(a) * IGn(a) - lambda * cost(a)
```

Why clip at zero: if removing an object increases downstream uncertainty, that
action should not gain reward from the entropy term.

## Fully Occluded Target

The target is missing from the visible graph, so the hidden target location is
modeled as a latent random variable:

```text
Z in C
C = { visible top-layer objects }
```

Here the VLM scores are mutually exclusive probabilities:

```text
s_i ~= p(Z = i | L, I)
sum_i s_i = 1
```

### Current Volume Proxy

`reason/invisible/geometry.py` computes a 2.5D proxy for each visible
candidate:

```text
A_i = visible mask area of object i
h_i = max(1, ground_depth - mean_depth_i)
```

If objects press on top of `i`, the code uses an equivalent area:

```text
Aeq_i = (A_i + sum_{k in predecessors(i)} A_k) / (1 + num_predecessors(i))
Vproxy_i = Aeq_i * h_i
```

Then:

```text
v_i = Vproxy_i / sum_j Vproxy_j
```

This is not a true physical volume. It is a useful "available hiding capacity"
proxy: a larger visible footprint and stronger depth-height signal imply a
larger occluded region where a missing target might be hidden.

The fused hidden-target belief is:

```text
P_0(i) = normalize_i(s_i^beta * v_i^gamma)
H_0 = H(P_0)
```

### Expected Information Gain

For action `a`, observation has two outcomes:

```text
hit:  target is found behind/under a, probability P_0(a)
miss: target is not found there, probability 1 - P_0(a)
```

If hit, residual entropy is zero. If miss, update the belief over the remaining
visible candidates. The Bayesian-only miss posterior would be:

```text
P_miss(j) = P_0(j) / (1 - P_0(a)), j != a
```

The current code instead recomputes a miss-side belief after removing `a`,
which is also defensible because the scene context changes:

```text
P_miss = posterior(C_-a | remove a and target not found)
H_miss = H(P_miss)
IG_full(a) = H_0 - (1 - P_0(a)) * H_miss
```

This is a standard expected entropy reduction objective. It is stronger
theoretically than the partial branch's structural IG because it is defined on
one latent variable, `Z`, and explicitly marginalizes over hit/miss outcomes.

Current utility:

```text
U_full(a) = IG_full(a)
U_full_graspability(a) = IG_full(a) * q_a
```

Recommended paper-ready form:

```text
IGn(a) = IG_full(a) / log2(max(2, |C|))
U_full(a) = q_a * IGn(a) - lambda * cost(a)
```

Do not multiply by `P_0(a)` again in the default full-occlusion score: `P_0(a)`
already appears inside `IG_full(a)` through the hit probability.

## Volume plus VLM Probability Model

The requested "use VLM and volume-derived probability p as the original
probability for deciding which object to grasp" can be written as:

```text
p_vlm(i) = normalized VLM probability or normalized VLM independent score
p_vol(i) = V_i / sum_j V_j
p_graph(i) = graph/path probability, if available

p0(i) = normalize_i(
    p_vlm(i)^beta *
    p_vol(i)^gamma *
    p_graph(i)^eta
)
```

For fully occluded targets:

```text
V_i = Vproxy_i = Aeq_i * h_i
score(i) = q_i * EIG(i; p0) - lambda * cost(i)
```

For partially occluded targets:

```text
V_i = A_i * h_i                         # optional local removability prior
p_graph(i) = bounded path probability i -> target
score(i) = q_i * p0(i) * IGn(i) - lambda * cost(i)
```

If a real 3D reconstruction is available later, replace the 2.5D proxy by a
voxel or point-cloud volume:

```text
V_i = sum_{x in Omega_i} voxel_volume * 1[x is unknown or target-hideable]
```

or by a TSDF/VGN-style grasp-aware volume:

```text
V_i = sum_{x in Omega_i} voxel_volume * hideability(x) * grasp_quality(x)
```

Then:

```text
p_vol(i) = V_i / sum_j V_j
```

This keeps the same probability pipeline while improving the physical meaning
of the volume term.

## Practical Recommendations

1. Keep Shannon entropy and expected information gain for `fully_occluded`.
   The derivation is clean: hidden target location is a latent random variable,
   and each removal action has hit/miss outcomes.
2. Keep entropy for `partially_occluded`, but name it "structural information
   gain" or "counterfactual entropy reduction" in writing. It compares current
   and residual subproblem uncertainty, not strict mutual information.
3. Replace partial path-sum geometry with the bounded union form when writing
   the final algorithm, or add it as a new ranking option.
4. Treat current invisible `area * height` as `Vproxy`, not real volume. This is
   defensible as a hidden-space capacity prior, but the notation should be
   honest.
5. Use power weights `beta`, `gamma`, and `eta` for calibration. VLM scores are
   not calibrated probabilities by default; exponent weights let experiments
   tune semantic vs. geometry vs. volume confidence.
6. Prefer branch-specific utilities:
   - partial: `q_i * p0(i) * IGn(i) - lambda * cost(i)`
   - fully occluded: `q_i * EIG(i; p0) - lambda * cost(i)`

## Minimal Algorithm Sketch

```text
Input: perception, target t, branch b
Output: next object to grasp/remove

1. Build candidate set C.
2. Get VLM score p_vlm over C.
3. Compute graph prior p_graph when target appears in the graph.
4. Compute volume prior p_vol when depth/masks are available.
5. Fuse p0 = normalize(p_vlm^beta * p_graph^eta * p_vol^gamma).
6. For each candidate a:
   if branch == partially_occluded:
      compute residual graph G_-a
      compute IG_struct(a) = H(p0) - H(posterior over residual candidates)
      score(a) = graspability(a) * p0(a) * IGn(a) - lambda * cost(a)
   if branch == fully_occluded:
      compute miss posterior after removing a
      compute EIG(a) = H(p0) - (1 - p0(a)) * H_miss(a)
      score(a) = graspability(a) * EIGn(a) - lambda * cost(a)
7. Return argmax_a score(a).
```
