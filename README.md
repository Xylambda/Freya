# FREYA

## Description
`Freya` is a deep learning framework built using numpy arrays. Is is just the code for [this post](https://quantdare.com/create-your-own-deep-learning-framework-using-numpy/) organized as as library. I do not plan to keep adding new features.

## Features
* Linear layers.
* DOC with external references.
* Step-by-step examples written in jupyter notebook.
* Common activation functions such as Sigmoid, ReLU and Hyperbolic tangent.
* Simple code to provide a simpler approach to neural network understanding.

## Install
```code
git clone https://github.com/Xylambda/Freya.git
pip install freya/.
```

## Developer mode
```code
git clone https://github.com/Xylambda/Freya.git
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

model.train(...)

model.predict(...)
```

You will find some use cases in `examples` folders.
