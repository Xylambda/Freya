# FREYA

## Description
`Freya` is a deep learning framework built using numpy arrays. It started as a 
learning project and I plan to keep adding new features to keep learning.

## Features
* Linear layers.
* DOC with external references.
* Step-by-step examples written in jupyter notebook.
* Common activation functions such as Sigmoid, ReLU and Hyperbolic tangent.
* Simple code to provide a simpler approach to neural network understanding.

## Install
```code
git clone
pip install freya/.
```

## Developer mode
```code
git clone
pip install -e -r requirements.txt freya/.
```

## How to use
`Freya` pretty much follows `Keras` philosophy for building neural networks.
```python
model = Model()
model.add(Linear(...))
model.add(ReLU(...))

model.add(Linear(...))
model.add(Sigmoid(...))

model.compile(...)
model.train(...)

model.predict(...)
```

You will find some use cases in `examples` folders.

## TODO
* Add tests.
* Recurrent layers.
* Batch size option.
* Add more loss functions.
* Add more activation functions.
* Management of split data (train and test).
* Add initializers (currently, only Kaiming is available).
* Add optimizers (currently, only vanilla Gradient Descent is available).