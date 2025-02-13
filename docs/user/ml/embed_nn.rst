********************************************
Embedding a Trained Neural Network in GAMSPy
********************************************


One of the most exciting applications of combining machine learning with
optimization is embedding a trained neural network into your optimization model.

Why?
====

You might need to verify that your neural network is robust. An activate research
field is using black-box solvers and getting certified minimum disturbations in
your input that confuses the neural network. Or, you might need a neural network
to approximate a complicated function for your from the data.


Sample Problem
==============


.. tabs::
   .. group-tab:: Feed-forward Neural Network

      Let’s assume you have trained a simple feed-forward neural network for optical
      character recognition, and you want to test the robustness of this neural network.
      For simplicity, we trained one for you on the MNIST dataset. If you
      want to follow this tutorial locally, you can download the `weights
      <https://github.com/GAMS-dev/gamspy/blob/develop/docs/_static/ffn_data.pth?raw=true>`_.

   .. group-tab:: Convolutional Neural Network

      Let’s assume you have trained a simple convolutional neural network for optical
      character recognition, and you want to test the robustness of this neural network.
      For simplicity, we trained one for you on the MNIST dataset. If you
      want to follow this tutorial locally, you can download the `weights
      <https://github.com/GAMS-dev/gamspy/blob/develop/docs/_static/cnn_data.pth?raw=true>`_.

.. image:: ../images/mnist.png
  :align: center

MNIST from:
LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
Proceedings of the IEEE 86.11 (1998): 2278-2324.


Importing the neural net
------------------------

.. tabs::
   .. group-tab:: Feed-forward Neural Network

     We trained the neural network using PyTorch, with a single hidden layer
     consisting of 20 neurons with ReLU activation function.

   .. group-tab:: Convolutional Neural Network

     We trained the convolutional neural network using PyTorch, with 2 hidden
     layers with ReLU activation function.

Let's start with the imports

.. code-block:: python

   import sys

   import gamspy as gp
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from gamspy.math.matrix import dim
   from torchvision import datasets, transforms

And then import the Neural Network, so we can get its weights:

.. tabs::
   .. group-tab:: Feed-forward Neural Network

      .. code-block:: python

         hidden_layer_neurons = 20

         class SimpleModel(nn.Module):
             def __init__(self):
                 super().__init__()
                 self.l1 = nn.Linear(784, hidden_layer_neurons, bias=True)
                 self.activation = nn.ReLU()
                 self.l2 = nn.Linear(hidden_layer_neurons, 10, bias=True)

             def forward(self, x):
                 x = torch.reshape(x, (x.shape[0], -1))
                 x = self.l1(x)
                 x = self.activation(x)
                 out = self.l2(x)
                 return out

         network = SimpleModel()
         network.load_state_dict(torch.load("ffn_data.pth", weights_only=True))

   .. group-tab:: Convolutional Neural Network

      .. code-block:: python

         class ConvNet(nn.Module):
             def __init__(self):
                 super(ConvNet, self).__init__()
                 self.l1 = nn.Sequential(
                     nn.Conv2d(1, 8, kernel_size=5, stride=2),
                     nn.ReLU()
                 )
                 self.l2 = nn.Sequential(
                     nn.Conv2d(8, 4, kernel_size=5, stride=2),
                     nn.ReLU(),
                 )
                 self.fc = nn.Linear(64, 10)
         
             def forward(self, x):
                 out = self.l1(x)
                 out = self.l2(out)
                 out = out.reshape(out.size(0), -1)
                 out = self.fc(out)
                 return out

         network = ConvNet()
         network.load_state_dict(torch.load("cnn_data.pth", weights_only=True))


To test the network's robustness, we will use an image from MNIST and
find the minimum change required for the neural network to misclassify it.


Testing with a sample
---------------------

.. code-block:: python

   mean = (0.1307,)
   std = (0.3081,)

   transform = transforms.Compose([transforms.ToTensor()])
   dataset1 = datasets.MNIST('../data', train=False, download=True, transform=transform)
   test_loader = torch.utils.data.DataLoader(dataset1)

   for data, target in test_loader:
       data, target = data, target
       break

   single_image = data[0]
   single_target = target[0]

   if torch.argmax(network(single_image)) == single_target:
       print("Model currently classifies correctly")
   else:
       print("Pick some other data")


Embedding the Neural Net
------------------------

Let's create the container and recreate the sample image
in GAMSPy for later use.

.. tabs::
   .. group-tab:: Feed-forward Neural Network

      In the feed-forward neural net, we pass the flattened image.

      .. code-block:: python

         m = gp.Container()
      
         image_data = single_image.numpy().reshape(784)
         image_target = single_target.numpy()
      
         image = gp.Parameter(m, name="image", domain=dim(image_data.shape), records=image_data)

   .. group-tab:: Convolutional Neural Network

      In the CNN, we do not flatten the image. Expected shape is Batch x Channel x Height x Width
       
      .. code-block:: python

         m = gp.Container()
       
         image_data = single_image.numpy().reshape(1, 1, 28, 28)
         image_target = single_target.numpy()
       
         image = gp.Parameter(m, name="image", domain=dim(image_data.shape), records=image_data)



Get the weights
---------------

Then we start creating GAMSPy parameters contaning the weights from the neural network:


.. tabs::
   .. group-tab:: Feed-forward Neural Network
      
      .. tabs::
         .. group-tab:: Linear Formulation
      
            Linear formulation will create the parameters for you. 
      
            .. code-block:: python
            
               # Let's get weights to numpy arrays
               l1_weight = network.l1.weight.detach().numpy()
               b1_weight = network.l1.bias.detach().numpy()
            
               l2_weight = network.l2.weight.detach().numpy()
               b2_weight = network.l2.bias.detach().numpy()
            
               l1 = gp.formulations.Linear(m, 784, 20, bias=True)
               l1.load_weights(l1_weight, b1_weight)
      
               l2 = gp.formulations.Linear(m, 20, 10, bias=True)
               l2.load_weights(l2_weight, b2_weight)
            
      
         .. group-tab:: Matrix Multiplication
      
            You need to create `w1`, `b1`, `w2` and `b2` parameters. 
      
            .. code-block:: python
      
               # Let's get weights to numpy arrays
               l1_weight = network.l1.weight.detach().numpy()
               b1_weight = network.l1.bias.detach().numpy()
            
               l2_weight = network.l2.weight.detach().numpy()
               b2_weight = network.l2.bias.detach().numpy()
      
               w1 = gp.Parameter(m, name="w1", domain=dim(l1_weight.shape), records=l1_weight)
               b1 = gp.Parameter(m, name="b1", domain=dim(b1_weight.shape), records=b1_weight)
            
               w2 = gp.Parameter(m, name="w2", domain=dim(l2_weight.shape), records=l2_weight)
               b2 = gp.Parameter(m, name="b2", domain=dim(b2_weight.shape), records=b2_weight)
      
      
      `w1` is a :math:`20 \times 784` matrix, `b1` is a vector of size :math:`20`,
      `w2` is a :math:`10 \times 20` matrix, and `b2` is a vector of size :math:`10`.
      The `image` is a vector of length `784`, which is obtained by flattening a
      :math:`28 \times 28` pixel image. Our task is to define the forward propagation
      process, where the `784` pixels are first mapped into :math:`\mathcal{R}^{20}`
      and then further mapped into :math:`\mathcal{R}^{10}`. In the final layer, we
      could apply the softmax function to obtain probabilities. However, we
      chose to work directly with the logits, as softmax is a monotonically
      increasing function.

   .. group-tab:: Convolutional Neural Network
      
      Technically, you can write convolutions without formulations but it is kind of
      messy to write. And using formulations over explicitly writing is suggested.

      .. code-block:: python

         # Let's get weights to numpy arrays
         # First layer: Conv2d(1, 8, kernel_size=5, stride=2)
         l1_weight = network.l1[0].weight.detach().numpy()  # Shape: (8, 1, 5, 5)
         b1_weight = network.l1[0].bias.detach().numpy()    # Shape: (8,)
         
         # Second layer: Conv2d(8, 4, kernel_size=5, stride=2)
         l2_weight = network.l2[0].weight.detach().numpy()  # Shape: (4, 8, 5, 5)
         b2_weight = network.l2[0].bias.detach().numpy()    # Shape: (4,)

         fc_weight = network.fc.weight.detach().numpy()
         fc_bias = network.fc.bias.detach().numpy()

         conv1 = gp.formulations.Conv2d(m, 1, 8, 5, stride=2)
         conv1.load_weights(l1_weight, b1_weight)
         
         conv2 = gp.formulations.Conv2d(m, 8, 4, 5, stride=2)
         conv2.load_weights(l2_weight, b2_weight)
         
         lin1 = gp.formulations.Linear(m, 64, 10)
         lin1.load_weights(fc_weight, fc_bias)


Create the input variable
-------------------------

We create a new variable called `noise`, which will be used to perturb the
input image. The `noise` variable has the same dimensions as the input image.
The variable `a1` will serve as the input to the neural network. 

.. tabs::

   .. group-tab:: Feed-forward Neural Network

      .. code-block:: python

         noise = gp.Variable(m, name="noise", domain=dim([784]))
         a1 = gp.Variable(m, name="a1", domain=dim([784]))

   .. group-tab:: Convolutional Neural Network

      .. code-block:: python

         noise = gp.Variable(m, name="noise", domain=dim([1, 1, 28, 28]))
         a1 = gp.Variable(m, name="a1", domain=dim([1, 1, 28, 28]))

It is defined by the `set_a1` equation, where the `noise` is added to the
image, followed by normalization, as the network was trained with normalized
inputs. We then ensure that `a1` stays within the valid range, so that the
`noise` cannot change any pixel to a negative value or exceed a value of 1.

.. code-block:: python

   set_a1 = gp.Equation(m, "set_a1", domain=a1.shape)
   set_a1[...] = a1 == (image + noise - mean[0]) / std[0]

   #set lower and upper bounds
   a1.lo[...] =   - mean[0] / std[0]
   a1.up[...] = (1 - mean[0]) / std[0]


Create the intermediate variables
---------------------------------

This step is only required if you do not use formulations.

.. tabs::
   .. group-tab:: Feed-forward Neural Network
      If you do not use formulations, you need to explicitly define the
      intermediate variables.

      .. tabs::
         .. group-tab:: Linear Formulation
            `z2` and `z3` will be created by linear formulations.
      
            .. code-block:: python
      
               # z2 = gp.Variable(m, name="a2", domain=dim([hidden_layer_neurons]))
               # z3 = gp.Variable(m, name="a3", domain=dim([10]))
      
         .. group-tab:: Matrix Multiplication
            We create `z2` and `z3`.
      
            .. code-block:: python
      
               z2 = gp.Variable(m, name="a2", domain=dim([hidden_layer_neurons]))
               z3 = gp.Variable(m, name="a3", domain=dim([10]))

   .. group-tab:: Convolutional Neural Network
      
      Formulations that we use, will create intermediate variables
      automatically.


Forward Pass
------------

Let's mimic the forward pass.

.. tabs::
   .. group-tab:: Feed-forward Neural Network
      .. tabs::
         .. group-tab:: Linear Formulation
      
            .. code-block:: python
      
               z2, _ = l1(a1)
               a2, _ = gp.math.relu_with_binary_var(z2)
      
            Then `z2` is created as output of the linear operation. Finally, we apply the
            :meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>` to obtain `a2`.
      
            Similarly, `z3` is created by the second linear operation `l2`:
      
            .. code-block:: python
      
               z3, _ = l2(a2)
      
         .. group-tab:: Matrix Multiplication
            .. code-block:: python
      
               forward_1 = gp.Equation(m, "eq2", domain=dim([hidden_layer_neurons]))
               forward_1[...] = z2 == w1 @ a1 + b1
      
               a2, _ = gp.math.relu_with_binary_var(z2)
      
            We define `z2` as the matrix multiplication of the weights and the previous
            layer, plus the bias term. Note that we use
            :meth:`relu_with_binary_var <gamspy.math.relu_with_binary_var>`
            to declare the `a2` variable, which automatically creates the necessary
            constraints and the activated variable for us.
      
            Similarly, we can define `z3`:
      
            .. code-block:: python
      
               forward_2 = gp.Equation(m, "eq3", domain=dim([10]))
               forward_2[...] = z3 == w2 @ a2 + b2


   .. group-tab:: Convolutional Neural Network
      
      .. code-block:: python

         z2, eqs1 = conv1(a1)
         a2, eqs2 = gp.math.relu_with_binary_var(z2)
         
         z3, eqs3 = conv2(a2)
         a3, eqs4 = gp.math.relu_with_binary_var(z3)
         
         z4, eqs5 = gp.formulations.flatten_dims(a3, [0, 1, 2, 3])
         z5, eqs6 = lin1(z4)

      We pass the input through the formulations, just like we would do it
      in PyTorch. `z2` is the output of the first convolution, `a2` is its
      activated version. And so on. Before passing input to the final linear
      layer, we flatten its dimensions as Linear formulations expect a certain
      shape.

This essentially completes the embedding of the neural network into our
optimization problem. If we were particularly interested in obtaining real
probabilities, we could have used `softmax` but for verification problem
it is not required. It also makes problem harder since it is a non-linear
function.


Additional Constraints
----------------------

Next, we define the component that specifies the adversarial attack. Our goal
is to make the model confuse our digit with another digit while making the
minimal possible change. We select a digit that is not the real digit for that.

We write the equation that forces another digit to be more likely than the
correct one.

In this example, the real digit is 7. We want the network to confuse it with 2.

.. tabs::
   .. group-tab:: Feed-forward Neural Network
      `z3` is the output of the feed-forward neural net.

      .. code-block:: python
      
         favor_confused = gp.Equation(m, "favor_confused")
         favor_confused[...] = z3["2"] >= z3["7"] + 0.1

   .. group-tab:: Convolutional Neural Network
      `z5` is the output of the CNN.

      .. code-block:: python
      
         favor_confused = gp.Equation(m, "favor_confused")
         favor_confused[...] = z5["2"] >= z5["7"] + 0.1


Confusing the neural network by completely changing the image would be trivial.
We aim for the minimum possible change to the original image. Therefore, we
define our objective as the L1 norm of the perturbations.

.. code-block:: python

   obj = gp.Variable(m, name="z")

   noise_upper = gp.Variable(m, name="noise_upper", domain=noise.domain)

   set_noise_upper_1 = gp.Equation(m, "set_noise_upper_1", domain=noise.domain)
   set_noise_upper_1[...] = noise_upper[...] >= noise

   set_noise_upper_2 = gp.Equation(m, "set_noise_upper_2", domain=noise.domain)
   set_noise_upper_2[...] = noise_upper[...] >= -noise

   set_obj = gp.Equation(m, "eq6")
   set_obj[...] = obj == gp.Sum(noise_upper.domain, noise_upper)


Finally, bringing it all together:

.. code-block:: python

   model = gp.Model(
       m,
       "min_noise",
       equations=m.getEquations(),
       objective=obj,
       sense="min",
       problem="MIP"
   )

   model.solve(output=sys.stdout, solver="cplex")


This takes a couple of seconds to solve, after which we can investigate:


.. tabs::
   .. group-tab:: Feed-forward Neural Network

      .. code-block:: python
      
         z3.toDense()
      
      .. code-block::
      
         [ -7.45149396 -13.61945982   0.77687953   2.1609202  -16.85390135
           -4.49846799 -22.13348944   0.67687953  -1.63533975  -8.07978064]

   .. group-tab:: Convolutional Neural Network

      .. code-block:: python
      
         z5.toDense()

      .. code-block::

         [ 1.13118789 -6.38256229  8.88743758  2.12709202 -9.48090804 -3.62381486
          -9.33158587  8.78743758 -0.89385071 -0.59244924]
      

You can see that the model assigned a higher likelihood to digit 2 than digit 7.
However, it's always beneficial to visually inspect the perturbed image and
verify that the network indeed misclassifies it.


.. tabs::
   .. group-tab:: Feed-forward Neural Network

      .. code-block:: python
      
         noise_data = noise.toDense()
      
         nn_input = torch.Tensor((noise_data + image_data - mean[0]) / std[0]).reshape(1, 784)
         print(network(nn_input))
      
      .. code-block::
      
         tensor([[ -7.4515, -13.6195,   0.7769,   2.1609, -16.8539,  -4.4985, -22.1335,
                    0.6769,  -1.6353,  -8.0798]], grad_fn=<AddmmBackward0>)


   .. group-tab:: Convolutional Neural Network

      .. code-block:: python
      
         noise_data = noise.toDense()
      
         nn_input = torch.Tensor((noise_data + image_data - mean[0]) / std[0]).reshape(1, 784)
         print(network(nn_input))
      
      .. code-block::
      
         tensor([[ 1.1315, -6.3826,  8.8873,  2.1269, -9.4810, -3.6241, -9.3314,  8.7875,
                  -0.8941, -0.5923]], grad_fn=<AddmmBackward0>)


You can see that, in FFN example, the largest logit in the last layer
corresponds to digit 3, confirming that our neural network is indeed
misclassifying the new image. You might say, didn't we target for digit 2?
But we just said that classifying it as 7 should be less likely than 2. Those
two statements can be true at the same time. That's why the objective that
is picked matters a lot in the context.

But the question remains: would we also confuse the image?


.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.cm as cm

   draw_nn = noise_data + image_data
   plt.imshow(draw_nn.reshape(28, 28), cmap='binary', vmin=0, vmax=1)

.. tabs::
   .. group-tab:: Feed-forward Neural Network
      
      .. image:: ../images/noisy_image.png
        :align: center
      
      A human would easily recognize this digit as a 7, not a 3, leading us to
      conclude that this network lacks robustness.

   .. group-tab:: Convolutional Neural Network

      .. image:: ../images/noisy_image_2.png
        :align: center
      
      A human would easily recognize this digit as a 7, not a 2, leading us to
      conclude that this network lacks robustness.

We demonstrated how easily a trained neural network can be embedded in GAMSPy.
Since GAMSPy supports a wide range of solvers, you're not limited to specific
activation functions. For instance, we could have used `tanh` as the activation
function and employed a nonlinear solver to find the minimum change, requiring
just two lines of code modification. More importantly, we've shown that writing
forward propagation in GAMSPy closely resembles how you would write it on paper.
