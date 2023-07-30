# Introduction to Policy Methods

## Lay of the Land

Optimise the Policy directly, without computing value function.

Topics:
- Finite Difference Policy Gradients: Use finite differences across time.
- Monte-Carlo Policy Gradient: Follows a randomly sampled trajectory and can move to directions to make it's policy better.
- Actor-Critic Policy Gradient: Combining the methods above with the value function methods.

We used to generate the policy using value function approximation
$$ V_\theta(s) = V^\pi(s)\\Q_\theta(s, a) = Q^\pi(s, a) $$

We have a parameterised policy function, where we can change the parameters to improve the policy by changing the distribution from which it's sampled. Adjusting $\theta$ to maximise net reward is the goal.
$$ \pi_\theta(s) = \mathbb{P}[a \vert s, \theta] $$

## Why is this better?

### Comparative of Value and Policy Methods

Advantages:
- Easier to represent the policy compactly, e.g. in Atari games move direction is easier than expected number of points in the future moves.
- Better convergence, more stable, value function can often oscillate or explode or fail to converge.
- Maximising over all actions can be prohibitively expensive, e.g. in continuous action spaces, or combinatorial. Adjust only parameters $\theta$.
- Learn stochastic policies, e.g. playing rock-paper-scissors needs stochastic behaviour.

Disadvantages:
- Hard to evaluate policy, evaluation is stochastic.
- Maybe slower to converge, since here small gradient based steps are taken whereas value function maximises in one go.
- It may converge to local optimal instead of the global optimal/
- The contrary of the point above, Value function maybe more compact.

### Example of Stochastic Policy doing better

![Untitled](images/reinforcement-learning/policy-gradients/stochastic-policy-example.png)

If we can measure only features of the current state, i.e. it's neighbors, then the two gray cells are identical. So it's not possible to play well here, you always go west or east in the gray squares, so reaching treasure is bad, and you just oscillate (between the left 2 squares) being given a deterministic policy. Stochastic solves the issue.

For Markov Decision Process, i.e. Fully Observed, then Deterministic Optimal Policy exists. But this is for POMDP or optimising on Parameters.

# How to do Policy Gradients

## Formulating the Task

Episodic environments can choose to optimize over start value assuming optimal behaviour then onwards.

$$ J_{1}(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}[v_1] $$

Formulation Based on average value of the states, for continuous environments, i.e. always be happy in life.

$$ J_{avV}(\theta) = \sum_s d^{\pi_\theta}(s) V^{\pi_\theta}(s) $$

Average reward per timestep formulation, maximizes the expected reward under the policy.

$$ J_{avR}(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(s, a) \mathcal{R}_s^a $$

All formulations use equivalent maximization approach.

## Getting the Optimizers

The following approaches do not use gradients, we have to wait the whole lifetime of our robot to get a single number out of it, i.e. how well it did in it's life.

- Hill Climbing
- Simplex / Amoeba / Nelder Mead
- Genetic Algorithms

The following are gradient based approaches, we use the trajectory followed by the robot, i.e. states and rewards it sees to optimize policy.

- Gradient Descent
- Conjugate Gradient
- Quasi-Newton

## Policy Gradients

$$ \Delta \theta = \alpha \nabla_\theta J(\theta) = \begin{bmatrix} \frac{\partial J(\theta)}{\partial \theta_1} \\ \frac{\partial J(\theta)}{\partial \theta_2} \\ \frac{\partial J(\theta)}{\partial \theta_3} \\ \ldots \end{bmatrix} $$

To optimise just compute the gradient of objective with respect to $\theta$, which is easier said than done. Since the derivative may be very high dimensional.

Approach 1: Perturb along each axis and compute, e.g. for robot walk with 12 axes of motions, do perturbations along each axis by doing 12 robot walks for each update.

![robot-policy-grad](images/reinforcement-learning/policy-gradients/robot-policy-grad.png)

Approach 2: Use some stochastic derivative approach, still only use perturbations, but

Approach 3: Compute it analytically if it's possible, this is the heart of feasible Policy Gradients.

## How to get a Score Function

Differentiating the policy, we will then want to take expectation of the derivative in the policy. So given the following log-likelihood trick, the computation is now easy, since

$$ \nabla_\theta \pi_\theta(s, a) = \pi_\theta(s, a) \frac{\nabla_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)} \\= \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) $$

Score of the task, therefore, will be:

$$ \nabla_\theta \log \pi_\theta(s, a) $$

### Softmax Policy Example

You can take softmax of independently weighted features, so the $\phi$ is the feature vector.

$$ \pi_\theta(s, a) \propto e^{\phi(s, a)^T \theta} $$

The score function now is

$$ \nabla_\theta \log \pi_\theta(s, a) = \phi(s, a) - \mathbb{E}_{\pi_\theta}[\phi(s, .)] $$

### Gaussian Policy

For continuous action space, the action is centred around a mean and has some variance.

$$ a \leftarrow \mathcal{N}(\mu(s), \sigma^2)\;\;\text{where}\;\mu(s) = \phi(s)^T \theta $$

So the score function now becomes:

$$ \nabla_\theta \log \pi_\theta(s, a) = \frac{(a - \mu(s)) \phi(s)}{\sigma^2} $$

## Derive Policy Gradients for General Case

### One Step MDP Derivation

$$ J(\theta) = \mathbb{E}_{\pi_\theta} [r] = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \mathcal{R}_{s, a} $$

So the derivative, using the log-likelihood trick, that gives the expectation of a gradient back as an expectation is the following:

$$ \nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) \mathcal{R}_{s, a} \\ = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) r] $$

### Multi Step MDP Derivation

💡 Policy Gradients Theorem states that for any of the objective optimisations, i.e. initial state max, average state value max, or average reward max, we get the same update formula: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)]$

This is the same as substituting the value function instead of the reward in the single step problem, so we are using the long term reward.

## Reinforce Algorithm

Let's use stochastic gradient ascent to do this, use the returned $v_t$ as an unbiased estimate of $Q^{\pi_\theta} (s_t, a_t)$.

![algo-reinforce](images/reinforcement-learning/policy-gradients/reinforce-algo.png)

# Actor Critic Method

## What's the Idea?

The Q values are very high variance, each simulated run will give something else.

We want to use a neural network, i.e. the "**Critic**", which will estimate the value function to reduce this variance.

## How to train the Critic

To evaluate the policy, we can use:

- Monte Carlo Policy evaluation
- Temporal Difference Learning
- TD($\lambda$) - Hybrid of the methods above

## Algorithm Description

### Action-Value ($Q$) Actor Critic

**Critic**: Updates $w$ with Linear TD(0)

**Actor**: Updates $\theta$ by policy gradients.

![algo-q-actor-critic](images/reinforcement-learning/policy-gradients/q-actor-critic-algo.png)

We no longer wait till the end of the episode. We use TD error to train the Critic.