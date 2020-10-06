# Simple MNIST classifier using numpy on CPU

## Fancy Graphic
![Image of Cost Function](/img/L1_900-->L2_300-->L3_10_e:500_a:0.04_acc:86.3%.png)

## Implements
- Basic Forward Propagation for user-defined network  
- Back Gradient Descent (no mini-batches, yet)  
- Basic Backpropagation and weight updates
- Various Activation functions  
- L2 Regularization
- Gradient Check for every parameter (so slow on CPU not worthwhile)
- Cost function and training accuracy plotting and auto-save  


## Terminating this project because...
- I'm making a new version which runs on GPU with [Cupy](https://github.com/cupy/cupy) [here](https://github.com/WyattJordan/cupy_nn), and I don't feel like supporting both CPU and GPU operations.

## [New GPU Version](https://github.com/WyattJordan/cupy_nn)
**Features will include:**  
- Mini-batch learning  
- Drop-Out Regularization
- Momentum Optimization
- RMSprop  
- Adaptive Momentum (Adam) Optimization
- Batch normalization
- Auto hyperparameter tuning
- and more!

## Resources
### Deep Learning Specialization with Andrew Ng from Coursera
- Week 1 - [Neural Networks and Deep Learning](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
- Week 2 - [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
- Week 3 - [Structuring Machine Learning Projects](https://www.youtube.com/playlist?list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b)
- Week 4 - [Convolutional Neural Networks](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
- Week 5 - [Sequence Models](https://www.youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6)  
### Michael Nielsen Online Neural Networks Book
- [Using Neural Nets on the MNIST](http://neuralnetworksanddeeplearning.com/chap1.html)
- [How backpropagation works](http://neuralnetworksanddeeplearning.com/chap2.html)
