# Train A Very Simple Model

This is the smallest useful training example in the repo:

- file: [src/trainSimpleModel.js](/C:/git/ai_trainer/src/trainSimpleModel.js)
- command: `npm run train:simple`

It trains a **linear model** from scratch in plain Node.js.

## What the model is

The model predicts a score from two inputs:

- `hoursStudied`
- `projectsBuilt`

Formula:

```txt
prediction = w1 * hoursStudied + w2 * projectsBuilt + bias
```

The trainable parts are:

- `w1`
- `w2`
- `bias`

That is all.

## What training means

Training is just repeating this loop:

1. Make predictions with the current weights.
2. Compare predictions to the true targets.
3. Measure how wrong the model is with a loss function.
4. Compute gradients.
5. Update the weights a little in the direction that reduces loss.

In this example:

- prediction function: linear equation
- loss function: mean squared error
- optimizer: basic gradient descent

## Why this is worth learning first

This gives you the real mechanics behind most ML systems:

- parameters
- forward pass
- loss
- gradients
- optimization

Later models are bigger and more nonlinear, but the loop is still the same shape.

## What to look for when you run it

The script prints:

- the loss shrinking over epochs
- the learned weights
- predictions on the training data
- predictions on a few unseen examples

If you understand why the loss goes down and why the weights settle where they do, you understand the foundation of training.

## Natural next step

After this, the next clean upgrade is:

1. binary classification
2. sigmoid activation
3. logistic loss

That gives you a tiny classifier instead of a regressor.
