## Methods

A novel transformer was developed to increase prediction speed and accuracy during real time optimization. The architecture is like that of a transformer used for neural machine translation but with the following modifications.
- The input to the decoder in neural machine translation is usually its own output. In this model, the optimizers manipulated variables and predicted MW are the inputs to the decoder.
- A global Embedding layer, inspired by vision transformers, was added to the attention mechanism to allow the model to easily draw connections between variables across time steps (Ramachandran et al. 2019).

The novel transformer architecture is shown in Figure 1 and the attention architecture is shown in figure 2

<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/transformer%20new%20poster%20-%20Copy.svg" width="500"/><img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/TSV%20attention%20-%20Copy.svg" width="250"/>

## Results and Discussion
The new transformer was compared with several kinds of recurrent neural networks (RNNs) as well as a time series transformer architecture developed by Dr. Park (Park 2022). All model architectures were optimized with Bayesian optimization. The model showed improvement over conventional RNN models as well as traditional time series transformers. The performance results can be seen in Table 1.

![image](https://github.com/ethan-r-gallup/TSCompare/assets/80715482/faf0558c-9792-4cc5-8a17-98d42cdfcdb1)

Even though the new model outperforms the current architecture, there are signs of issues that need to be addressed. Evaluating the input gradients reveals that the model has become overly dependent on the predicted gross MW and the end remote MW. This will cause issues in deployment because those values are always estimated and are often wrong. This is shown in Figure 3.

<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/GRU%20mean%20input%20gradients.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/GRU%20input%20gradients%20variance.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/GRU%20max%20input%20gradients.svg" width="3000"/>

<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Park%20Transformer%20encoder%20mean%20input%20gradients.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Park%20Transformer%20encoder%20input%20gradient%20variance.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Park%20Transformer%20encoder%20max%20input%20gradients.svg" width="3000"/>

<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Transformer%20encoder%20mean%20input%20gradients.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Transformer%20encoder%20input%20gradient%20variance.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Transformer%20encoder%20max%20input%20gradients.svg" width="3000"/>

<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Transformer%20decoder%20mean%20input%20gradients.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Transformer%20decoder%20input%20gradient%20variance.svg" width="3000"/>
<img src="https://github.com/ethan-r-gallup/TSCompare/blob/main/figures/Transformer%20decoder%20max%20input%20gradients.svg" width="3000"/>

This problem can be addressed relatively easily by applying a dropout mask to the input layer that replaces a percentage of these values with either zeros or random values to lower in model’s dependence on these inputs. After that, the architecture will be optimized again and should be ready for deployment.

Another interesting note is that the traditional time series transformer developed by Dr. Park does not have a decoder. This increases the computation speed but does not provide the ability to artificially weight the contributions of the optimizers control decisions.

## Pending Updates
- Finishing optimization of models trained with artificial noise.
- Rebuild data pipeline to pull testing data from the same dataset as training data and split it in the code.
- Add results from encoder-decoder recurrent models.
- Permutation importance for the inputs of each model.
- Closed loop demo.
- Data for control performance on plant.

## References
Park J. (2022). Hybrid Machine Learning and Physics-Based Modeling Approaches for Process Control and Optimization (dissertation). 

Ramachandran P., Parmar N., Vaswani A., Bello I., Levskaya A, Shlens J. (2019). Stand-Alone Self-Attention in Vision Models, arXiv.
