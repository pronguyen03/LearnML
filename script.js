const tf = require("@tensorflow/tfjs");

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
// require("@tensorflow/tfjs-node");
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  loss: "meanSquaredError",
  optimizer: "sgd"
});

const xs = tf.tensor2d([1, 2, 3, 4, 5, 6, 7], [7, 1]);
const ys = tf.tensor2d([2, 4, 6, 6, 10, 12, 14], [7, 1]);
model.fit(xs, ys, { epochs: 500 }).then(() => {
  model.predict(tf.tensor2d([10], [1, 1])).print();
});
