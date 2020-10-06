import numpy as np
import dataloader
from layer import layer
from network import network
from plotter import plotter
from grad_check import gradcheck

def main():
    np.set_printoptions(threshold=np.inf)
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.load_data()

    # play around with # of layers, # of hidden units, activation types
    dims = [train_x.shape[0], 1568, 392, 98, 10]
    activations = ["relu","relu","relu","sigmoid"]
    net = network(dims, activations, train_x.shape[1])
    
    epochs = 50
    learning_rate = 0.04
    costs = []
    accs  = []
    checkgrad = False
    plot = True
    for e in range(epochs):
        #learning_rate = learning_rate / (1 + 0.08*e) # can decay if desired
        output = net.propagate_all(train_x)
        costs.append(net.compute_cost(output, train_y))
        accs.append (net.compute_acc( output, train_y))
        print("Epoch {} has acc {:.3f}% and cost {:.5f} with rate {:.4f}".format(e,accs[-1],costs[-1],learning_rate))

        net.backprop_all(output, train_y)
        if checkgrad:
            check_gradient(net) # uselessly slow
        net.update_all(learning_rate)

    if plot:
        plt = plotter(dims, costs, accs, learning_rate)
        plt.plot("and plain back gradient descent")

def check_gradient(net, train_x, train_y):
    print("---------- Starting gradient check... ----------")
    gcheck = gradcheck(net)
    check = gcheck.check_gradient(train_x, train_y) # requires dw and db from backprop
    gcheck.output_results()

if __name__ == "__main__":
    main()
        
