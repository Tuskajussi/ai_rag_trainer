const FEATURE_NAMES = ["hoursStudied", "projectsBuilt"];

const trainingData = [
  { features: [1, 0], target: 17 },
  { features: [1, 1], target: 25 },
  { features: [2, 1], target: 29 },
  { features: [2, 2], target: 36 },
  { features: [3, 1], target: 33 },
  { features: [3, 2], target: 40 },
  { features: [4, 2], target: 45 },
  { features: [4, 3], target: 53 },
  { features: [5, 2], target: 48 },
  { features: [5, 3], target: 57 },
  { features: [6, 3], target: 63 }
];

function dotProduct(left, right) {
  let total = 0;

  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }

  return total;
}

function predict(features, model) {
  return dotProduct(features, model.weights) + model.bias;
}

function meanSquaredError(samples, model) {
  let total = 0;

  for (const sample of samples) {
    const error = predict(sample.features, model) - sample.target;
    total += error * error;
  }

  return total / samples.length;
}

function computeGradients(samples, model) {
  const weightGradients = new Array(model.weights.length).fill(0);
  let biasGradient = 0;

  for (const sample of samples) {
    const prediction = predict(sample.features, model);
    const error = prediction - sample.target;

    for (let index = 0; index < sample.features.length; index += 1) {
      weightGradients[index] += (2 * error * sample.features[index]) / samples.length;
    }

    biasGradient += (2 * error) / samples.length;
  }

  return {
    biasGradient,
    weightGradients
  };
}

function trainLinearModel(samples, options = {}) {
  const epochs = options.epochs ?? 2_500;
  const learningRate = options.learningRate ?? 0.01;
  const logEvery = options.logEvery ?? 250;
  const model = {
    bias: 0,
    weights: new Array(samples[0].features.length).fill(0)
  };

  console.log("Training a simple linear model");
  console.log(`Samples: ${samples.length}`);
  console.log(`Features: ${FEATURE_NAMES.join(", ")}`);
  console.log(`Epochs: ${epochs}`);
  console.log(`Learning rate: ${learningRate}`);
  console.log("");
  console.log("Model form");
  console.log("prediction = w1 * hoursStudied + w2 * projectsBuilt + bias");
  console.log("");

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const { biasGradient, weightGradients } = computeGradients(samples, model);

    for (let index = 0; index < model.weights.length; index += 1) {
      model.weights[index] -= learningRate * weightGradients[index];
    }

    model.bias -= learningRate * biasGradient;

    if (epoch === 1 || epoch % logEvery === 0 || epoch === epochs) {
      const loss = meanSquaredError(samples, model);
      console.log(
        `epoch=${String(epoch).padStart(4, " ")} loss=${loss.toFixed(4)} weights=[${model.weights
          .map((value) => value.toFixed(4))
          .join(", ")}] bias=${model.bias.toFixed(4)}`
      );
    }
  }

  return model;
}

function explainModel(model) {
  console.log("");
  console.log("Learned model");
  for (let index = 0; index < model.weights.length; index += 1) {
    console.log(`${FEATURE_NAMES[index]} weight = ${model.weights[index].toFixed(4)}`);
  }
  console.log(`bias = ${model.bias.toFixed(4)}`);
}

function printPredictions(samples, model) {
  console.log("");
  console.log("Training set predictions");

  for (const sample of samples) {
    const prediction = predict(sample.features, model);
    const rounded = Math.round(prediction * 100) / 100;
    console.log(
      `${JSON.stringify(sample.features)} -> predicted=${rounded.toFixed(2)} actual=${sample.target.toFixed(2)}`
    );
  }
}

function printNewExamples(model) {
  const newExamples = [
    { features: [2, 3], label: "2 study hours, 3 projects" },
    { features: [4, 1], label: "4 study hours, 1 project" },
    { features: [6, 4], label: "6 study hours, 4 projects" }
  ];

  console.log("");
  console.log("New predictions");

  for (const sample of newExamples) {
    const prediction = predict(sample.features, model);
    console.log(`${sample.label} -> predicted score ${prediction.toFixed(2)}`);
  }
}

function main() {
  const model = trainLinearModel(trainingData, {
    epochs: 2_500,
    learningRate: 0.01,
    logEvery: 250
  });

  explainModel(model);
  printPredictions(trainingData, model);
  printNewExamples(model);

  console.log("");
  console.log("What just happened");
  console.log("- The model started with weights and bias at 0.");
  console.log("- Each epoch computed prediction errors on all samples.");
  console.log("- Those errors were converted into gradients.");
  console.log("- Gradient descent nudged the weights and bias to reduce loss.");
  console.log("- After enough updates, the line fit the data reasonably well.");
}

main();
