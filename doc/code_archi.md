## Code Architecture 
The code consists of two files: solver.py and main.py 
### solver.py
The most important class in this file is Solver_Grad_4DVarNN. All learnable parameters are under this class. It includes the following:
**Attributes**
- phi_r: CNN-based Gibbs-Energy (GENN) model.

- model_Grad: LSTM-based model whose role is to update the variational cost gradient. 
  
- model_H: The observation model. 

- Other Hyperparameters: OptimType, GradType, DimObs, NGrad....

**Methods**

- var_cost: Takes as input a current state x, an observation and a mask, and outputs the variational cost and its gradient w.r.t
    the current state.
 
- solve_step: Gets as input a current state, an observation and a mask, and gives as output the updated state x. 

- solve: Apllies solver_step NGrad times .

### main.py
This script contains three key parts: Data generation , phi_r and model_H architectures definition , a pytorch lightning module which takes into account the training and testing of the overall model. 

**Data generation**

At the end of this part of code, three dataloaders are generated: train_dataloader, val_dataloader, test_data_loader. 

**Architectures definition**

phi_r is an auto-encoder with 11 convolution-layers-encoder and model_H is a parameter-free model that outputs (x-yobs)mask.  

**Pytorch lightning module** 

This module automates training and testing. It has the following key attributes and methods: 

**Attributes**

- model: It is an instantiation of the solver Solver_Grad_4DVarNN. 
    
- m_Grad: It is an instantiation of the LSTM-based model.
- model_LR: Apply an average pooling.

**Methods**

- configure_optimisers -> Defines optimizers and  learning rate schedulers.

- training_step , validation_step, test_step -> Run on each training, validation and test batch respectively.  

- test_epoch_end -> At the end of the test step, this method is applied to save the model results on the test data.

- compute_loss -> Takes as input a batch and a step('train', 'val' or 'test'), and computes a loss and an output. 
