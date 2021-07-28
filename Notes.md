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


# Workflow
## QT console to get interactive plots
 * Start kernel on compute node:

``` from jeanzay
salloc --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread -C v100-16g --qos=qos_gpu-t3 -A yrf@gpu --time=05:00:00
ssh $(squeue -u $USER -h -o %R)  'cd $(pwd) && bash -c " . ~/.bashrc && conda activate --stack 4dvarnet && jupyter console --ip=0.0.0.0 -f=jz_kernel.json"'
```

```from laptop
# start sshuttle with 
scp jeanzay:scratch/<worktree>/jz_kernel.json .
jupyter qtconsole --ssh=jz-node --existing=/home/q20febvr/jz_kernel.json --ConsoleWidget.include_other_output=True
```
