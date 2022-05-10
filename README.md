# Residual Q-Network

Main codes adapted from [SMAC framework](https://github.com/starry-sky6688/StarCraft) to be used with the environments from the [MA-gym](https://github.com/koulanurag/ma-gym) collection. The original [SMAC framework](https://github.com/starry-sky6688/StarCraft) was used for the Starcraft environments. Acknowledgements also to [pymarl](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) for Starcraft environemnts.
<p>
Experiments can be run with a command like:
  
```python
python3 main.py --env <environment_name> --alg <algorithm_name> --n_epoch <number_of_epochs>
```
<p>
Examples:
<p>
  
```python
python3 main.py --env PredatorPrey7x7-v0 --alg rqn --n_epoch 20000
python3 main.py --env Switch2-v0 --alg qmix --n_epoch 20000
```


