use super::math;
use super::neuron;
use std::iter;

struct Network {
    hidden_layer: Vec<neuron::Neuron>,
    outputs: Vec<neuron::Neuron>,
    epochs: usize,
}

impl Network {
    const LEARN_RATE: f64 = 0.5;

    fn new(num_inputs: usize, num_hidden_layers: usize, num_outputs: usize) -> Network {
        let bias = 0 as f64;
        let hidden_layer: Vec<_> = iter::repeat_with(move || {
            let weights = vec![0.0; num_inputs];
            return neuron::Neuron {
                weights: weights,
                bias,
            };
        })
        .take(num_hidden_layers)
        .collect();

        let outputs: Vec<_> = iter::repeat_with(move || {
            let weights = vec![0.0; num_hidden_layers];
            return neuron::Neuron {
                weights: weights,
                bias,
            };
        })
        .take(num_outputs)
        .collect();

        return Network {
            hidden_layer,
            outputs,
            epochs: 1000,
        };
    }

    fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        let hidden_outs: Vec<_> = self
            .hidden_layer
            .iter()
            .map(|neuron| neuron.feed_forward(inputs))
            .collect();
        return self
            .outputs
            .as_slice()
            .iter()
            .map(|output| output.feed_forward(&hidden_outs))
            .collect();
    }

    fn train(&self, data: &[Vec<f64>], actuals: &[Vec<f64>]) {
        for epoch in 0..self.epochs {
            let mut sum_error = 0.0;
            for (entity, actual) in data.iter().zip(actuals.iter()) {
                let outputs: Vec<_> = self.feed_forward(entity.as_slice());
                sum_error += math::mse(&outputs, actual.as_slice());
            }
            println!(">epoch={}, error={}", epoch, sum_error);
        }
    }
}

#[test]
fn feed_forward() {
    let weights = vec![0.0, 1.0];
    let bias = 0.0;
    let inputs = [2.0, 3.0];

    let network = Network {
        hidden_layer: vec![
            neuron::Neuron {
                weights: vec![0.0, 1.0],
                bias,
            },
            neuron::Neuron {
                weights: vec![0.0, 1.0],
                bias,
            },
        ],
        outputs: vec![neuron::Neuron {
            weights: weights,
            bias,
        }],
        epochs: 1000,
    };

    assert_eq!(vec![0.7216325609518421], network.feed_forward(&inputs))
}

#[test]
fn feed_forward_two_outputs() {
    let inputs = [1.0, 0.0];

    let network = Network {
        hidden_layer: vec![neuron::Neuron {
            weights: vec![0.13436424411240122, 0.8474337369372327],
            bias: 0.763774618976614,
        }],
        outputs: vec![
            neuron::Neuron {
                weights: vec![0.2550690257394217],
                bias: 0.49543508709194095,
            },
            neuron::Neuron {
                weights: vec![0.4494910647887381],
                bias: 0.651592972722763,
            },
        ],
        epochs: 1000,
    };

    assert_eq!(
        vec![0.6629970129852887, 0.7253160725279748],
        network.feed_forward(&inputs)
    )
}

#[test]
fn train() {
    let network = Network {
        hidden_layer: vec![neuron::Neuron {
            weights: vec![0.13436424411240122, 0.8474337369372327],
            bias: 0.763774618976614,
        }],
        outputs: vec![
            neuron::Neuron {
                weights: vec![0.2550690257394217],
                bias: 0.49543508709194095,
            },
            neuron::Neuron {
                weights: vec![0.4494910647887381],
                bias: 0.651592972722763,
            },
        ],
        epochs: 20,
    };

    let dataset = vec![
        vec![2.7810836, 2.550537003],
        vec![1.465489372, 2.362125076],
        vec![3.396561688, 4.400293529],
        vec![1.38807019, 1.850220317],
        vec![3.06407232, 3.005305973],
        vec![7.627531214, 2.759262235],
        vec![5.332441248, 2.088626775],
        vec![6.922596716, 1.77106367],
        vec![8.675418651, -0.242068655],
        vec![7.673756466, 3.508563011],
    ];
    let actuals = [
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
    ];

    network.train(dataset.as_slice(), &actuals);
}
