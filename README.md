| Read me |
|---------|
|         |

To: Jake Tuttle

From: Ethan Gallup

Date: 7/21/2023

**Methods**

A novel transformer was developed to increase prediction speed and accuracy during real time optimization. The architecture is like that of a transformer used for neural machine translation but with the following modifications.

-   The input to the decoder in neural machine translation is usually its own output. In this model, the optimizers manipulated variables and predicted MW are the inputs to the decoder.
-   A global Embedding layer, inspired by vision transformers, was added to the attention mechanism to allow the model to easily draw connections between variables across time steps (Ramachandran et al. 2019).

![](media/7a54212ac142d5d736841ccae8ba7f29.png)The novel attention architecture is shown in Figure 1 and the transformer architecture is shown in figure 2.

![](media/b1ce77e9bedbf6f72fe235caf59b7d40.png)

**Results and Discussion**

![A table with numbers and a red green and white line Description automatically generated](media/f7e66e69a5e0def58570500370ac6d4b.png)The new transformer was compared with several kinds of recurrent neural networks (RNNs) as well as a time series transformer architecture developed by Dr. Park (Park 2022). All model architectures were optimized with Bayesian optimization. The model showed improvement over conventional RNN models as well as traditional time series transformers. The performance results can be seen in Table 1.

Even though the new model outperforms the current architecture, there are signs of issues that need to be addressed. Evaluating the input gradients reveals that the model has become overly dependent on the predicted gross MW and the end remote MW. This will cause issues in deployment because those values are always estimated and are often wrong. This is shown in Figure 3.

![A red and blue gradient Description automatically generated](media/cea28c8be7621aa279a2e60eec8d4be8.png)![](media/72b4217fa7c0d3d2a6460cf784fc3862.png)

![A graph of a number of numbers Description automatically generated](media/0c2bb1e5df8465e5d2bfb9e9f117ffde.png)

![](media/867b01f39f1a1790ab28fa201815877a.png)

This problem can be addressed relatively easily by applying a dropout mask to the input layer that replaces a percentage of these values with either zeros or random values to lower in model’s dependence on these inputs. After that, the architecture will be optimized again and should be ready for deployment.

Another interesting note is that the traditional time series transformer developed by Dr. Park does not have a decoder. This increases the computation speed but does not provide the ability to artificially weight the contributions of the optimizers control decisions.

**References**

Park J. (2022). Hybrid Machine Learning and Physics-Based Modeling

Approaches for Process Control and Optimization (dissertation).

Ramachandran P., Parmar N., Vaswani A., Bello I., Levskaya A, Shlens J. (2019).

Stand-Alone Self-Attention in Vision Models, arXiv.
