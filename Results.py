import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

from LoadData import * 
from NeuralNetwork import *

def main():
    X, y = load_data()
    model = neural_network(X,y)
    index = int(input("Please enter a random image numbered 0-4999 (The model will attempt to predict the digit): "))
    
    yhat = prediction(model, index, X)
    plot(X, y, model, index, yhat)


def plot(X, y, model, index, yhat):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
        
    X_reshaped = X[index].reshape((20,20)).T
        
    ax.imshow(X_reshaped, cmap='gray')
    ax.set_title(f"Prediction of Digit Below: {yhat}", fontsize=10)
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()