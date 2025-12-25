# Simple Neural Network from Scratch (NumPy)

This repository contains a minimal implementation of a single-layer neural network written in pure Python using **NumPy**, without any machine learning frameworks.

The project is intended for educational purposes and demonstrates the core principles behind neural networks: the **sigmoid activation function** and training via **backpropagation**.

## Features

- Pure NumPy implementation
- Single-layer neural network (logistic regression–like)
- Sigmoid activation function
- Manual weight initialization
- Backpropagation-based training
- Multiple independent datasets for demonstration

## How It Works

The model:
- Accepts an input vector of fixed size **3**
- Produces a single output value in the range **[0, 1]**
- Learns by adjusting weights based on prediction error

Each dataset is trained **independently**, resulting in different learned weights for different data patterns.

## Code Overview

- `sigmoid(x)`  
  Implements the sigmoid activation function.

- `train(X, y, epochs=20000)`  
  Trains the neural network using:
  - random weight initialization
  - forward propagation (`dot product + sigmoid`)
  - error calculation
  - backpropagation-based weight updates

- `X1, y1`, `X2, y2`, `X3, y3`  
  Three different datasets used to demonstrate learning on different patterns.

- `w1, w2, w3`  
  Trained weights for each dataset.

- Final predictions are computed using:
  ```python
  sigmoid(np.dot(input, weights))
  ```

## Example Output

After training, the script prints prediction results for each dataset:

```
Result Dataset 1:
0.01...

Result Dataset 2:
0.98...

Result Dataset 3:
0.02...
```


## Requirements

- Python 3.x
- NumPy

Install NumPy:

```
pip install numpy
```

## How to Run

Run the script using:

```
python examples_datasets.py
```


## Educational Purpose

This project is useful for:
- Understanding neural networks from first principles
- Learning how backpropagation works internally
- Demonstrating why different datasets require separate training
- Exploring machine learning without high-level abstractions

## Limitations

- No bias term
- No learning rate parameter
- Single neuron only
- No train/test split
- Not intended for production use

## License

MIT License.  
Free to use for learning, teaching, and experimentation.
