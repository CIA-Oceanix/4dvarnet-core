- [] training swot somme 2 comp


--------------------
Étape 1 : arriver à 0 erreur en utilisant des observation non bruitées sur la fauchée

- plusieurs variables d'observations dans l'état: O / N
- fauchée somme de 3 ou de 2 composantes: O / N


Étape 2 : impact loss rec globale


-----------------------

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


# Current Xps:
```
612827-> swot ; test 613421
lightning_logs/version_612827/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.08.ckpt
lightning_logs/version_612827/checkpoints/modelSLAInterpGF-Exp3-epoch=28-val_loss=0.08.ckpt
lightning_logs/version_612827/checkpoints/modelSLAInterpGF-Exp3-epoch=31-val_loss=0.08.ckpt
612828-> roll : test 613430
lightning_logs/version_612828/checkpoints/modelSLAInterpGF-Exp3-epoch=20-val_loss=0.09.ckpt
lightning_logs/version_612828/checkpoints/modelSLAInterpGF-Exp3-epoch=23-val_loss=0.09.ckpt
lightning_logs/version_612828/checkpoints/modelSLAInterpGF-Exp3-epoch=56-val_loss=0.08.ckpt
613264-> nad roll: test 613391
lightning_logs/version_613264/checkpoints/modelSLAInterpGF-Exp3-epoch=00-val_loss=0.13.ckpt

```
