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
            return neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: weights,
                bias,
            });
        })
        .take(num_hidden_layers)
        .collect();

        let outputs: Vec<_> = iter::repeat_with(move || {
            let weights = vec![0.0; num_hidden_layers];
            return neuron::Neuron::Output(neuron::NeuronProperties {
                weights: weights,
                bias,
            });
        })
        .take(num_outputs)
        .collect();

        return Network {
            hidden_layer,
            outputs,
            epochs: 1000,
        };
    }

    fn feed_forward_(&self, inputs: &[f64], layer: &[neuron::Neuron]) -> Vec<f64> {
        return layer.iter()
            .map(|neuron| neuron.feed_forward(inputs))
            .collect();
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

    // fn back_propogate_output(&self, neuron: &neuron::Neuron, preds: &[f64]) -> neuron::Neuron {
    //     let sum = math::dot_product(preds, neuron.weights.as_slice());
    //     let dsum = math::deriv_sigmoid(sum);
    //     let weights: Vec<_> = neuron.weights.iter().map(|w| w * dsum).collect();
    //     return neuron::Neuron { weights, bias: dsum };
    // }

    // fn back_propogate_hidden(&self, neuron: &neuron::Neuron, preds: &[f64]) -> neuron::Neuron {
    //     let sum = math::dot_product(preds, neuron.weights.as_slice());
    //     let dsum = math::deriv_sigmoid(sum);
    //     let weights: Vec<_> = preds.iter().map(|w| w * dsum).collect();
    //     return neuron::Neuron { weights, bias: dsum };
    // }

    fn back_propogate(&self, inputs: &[f64], expecteds: &[f64], pred_outputs: &[f64]) -> Vec<neuron::Neuron> {
        // let dlosses: Vec<_> = pred_outputs.iter().zip(expecteds.iter())
        //     .map(|(pred, expected)| -2.0 * (expected - pred))
        //     .collect();
        // let doutputs: Vec<_> = self.outputs.iter()
        //     .map(|output| {
        //         let preds: Vec<_> = self.hidden_layer.iter().map(|hidden| hidden.feed_forward(inputs)).collect();
        //         return self.back_propogate_output(output, preds.as_slice());
        //     })
        //     .collect();
        // let dhiddens: Vec<_> = self.hidden_layer.iter()
        //     .map(|hidden| self.back_propogate_hidden(hidden, inputs))
        //     .collect();

        // let deltas: Vec<_> = dlosses.iter()
        //     .map(|loss| {
        //         let weights: Vec<_> = doutputs.iter().zip(dhiddens.iter())
        //             .flat_map(|(output, hidden)| output.weights.iter().zip(hidden.weights.iter())
        //                 .map(|(wo, wh)| wo * wh * loss))
        //             .collect();
        //         let bias = 0.0 * loss;
        //         return neuron::Neuron { weights, bias };
        //     })
        //     .collect();
        // return deltas;
        return vec![];
    }

    fn train(&self, data: &[Vec<f64>], actuals: &[Vec<f64>]) {
        for epoch in 0..self.epochs {
            for (entity, actual) in data.iter().zip(actuals.iter()) {
                let new_outputs = self.feed_forward(entity.as_slice());
                let deltas = self.back_propogate(entity.as_slice(), actual.as_slice(), new_outputs.as_slice());
            }
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
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![0.0, 1.0],
                bias,
            }),
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![0.0, 1.0],
                bias,
            }),
        ],
        outputs: vec![neuron::Neuron::Output(neuron::NeuronProperties {
            weights: weights,
            bias,
        })],
        epochs: 1000,
    };

    assert_eq!(vec![0.7216325609518421], network.feed_forward(&inputs))
}

#[test]
fn feed_forward_two_outputs() {
    let inputs = [1.0, 0.0];

    let network = Network {
        hidden_layer: vec![
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![0.13436424411240122, 0.8474337369372327],
                bias: 0.763774618976614,
            })
        ],
        outputs: vec![
            neuron::Neuron::Output(neuron::NeuronProperties {
                weights: vec![0.2550690257394217],
                bias: 0.49543508709194095,
            }),
            neuron::Neuron::Output(neuron::NeuronProperties {
                weights: vec![0.4494910647887381],
                bias: 0.651592972722763,
            }),
        ],
        epochs: 1000,
    };

    assert_eq!(
        vec![0.6629970129852887, 0.7253160725279748],
        network.feed_forward(&inputs)
    )
}

#[test]
fn feed_forward_fixed_weights() {
    let inputs = [-2.0, -1.0];

    let network = Network {
        hidden_layer: vec![
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
        ],
        outputs: vec![
            neuron::Neuron::Output(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
        ],
        epochs: 1000,
    };

    assert_eq!(vec![0.5236951740839997], network.feed_forward(&inputs));
}

#[test]
fn back_propogate() {
    let inputs = [-2.0, -1.0];
    let actuals = [1.0];

    let network = Network {
        hidden_layer: vec![
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
        ],
        outputs: vec![
            neuron::Neuron::Output(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
        ],
        epochs: 1000,
    };

    // assert_eq!(
    //     vec![neuron::Neuron { weights: vec![0.021469535265811107, 0.010734767632905554], bias: -0.0 }],
    //     network.back_propogate(&inputs, &actuals, network.feed_forward(&inputs).as_slice())
    // );
}

#[test]
fn train() {
    let network = Network {
        hidden_layer: vec![
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
            neuron::Neuron::Hidden(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
        ],
        outputs: vec![
            neuron::Neuron::Output(neuron::NeuronProperties {
                weights: vec![1.0, 1.0],
                bias: 0.0,
            }),
        ],
        epochs: 20,
    };

    let dataset = [
        vec![-2.0, -1.0],
        vec![25.0, 6.0],
        vec![17.0, 4.0],
        vec![-15.0, -6.0]
    ];
    let actuals = [
        vec![1.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
    ];

    network.train(&dataset, &actuals);
}
