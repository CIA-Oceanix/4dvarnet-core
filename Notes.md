# Xp description

### Motivation
Using the 4dvarnet method to reconstruct the best possible ssh field on the swath

### Steps:

- [] [ Compute baseline metrics ](#metrics)
- [] Implement xp with
	- [] [the state with 4 components](#state)
	- [] [the cost specific to the reconstruction on the swath](#cost)

#### Metrics
 * Compute the mse on the grid for the points with swath data
 * Compute the MSE on the swath
 * Compute the MSE for the gradient
 * Add more noise

#### State
The state is made of 4 components:
 * Low res (OI with 4 nadirs)
 * obs with noise (nad, swot+roll)
 * anomaly of the complete field
 * anomaly on the swath

#### Cost
 * Add True obs field
 * Add a loss term computed only on the swath
