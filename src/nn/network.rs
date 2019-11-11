use super::neuron;
use super::neuron::{Neuron, NeuronBase};
use std::iter;

#[derive(Debug, PartialEq)]
struct Network {
    hidden_layer: Vec<neuron::HiddenNeuron>,
    outputs: Vec<neuron::OutputNeuron>,
}

impl Network {
    const LEARN_RATE: f64 = 0.1;

    pub fn new(num_inputs: usize, num_hidden_layers: usize, num_outputs: usize) -> Network {
        let bias = 0 as f64;
        let hidden_layer: Vec<_> = iter::repeat_with(move || {
            let weights = vec![1.0; num_inputs];
            return neuron::HiddenNeuron::new(weights, bias);
        })
        .take(num_hidden_layers)
        .collect();

        let outputs: Vec<_> = iter::repeat_with(move || {
            let weights = vec![1.0; num_hidden_layers];
            return neuron::OutputNeuron::new(weights, bias);
        })
        .take(num_outputs)
        .collect();

        return Network {
            hidden_layer,
            outputs,
        };
    }

    pub fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        let hidden_outs = self.feed_forward_neurons(&self.hidden_layer, inputs);
        return self.feed_forward_neurons(&self.outputs, &hidden_outs);
    }

    pub fn back_propogate(&self, inputs: &[f64], actuals: &[f64]) -> Self {
        let ypred = self.feed_forward(inputs);
        let dLs: Vec<_> = ypred.iter().zip(actuals.iter()).map(|(input, actual)| -2.0 * (actual - input)).collect();
        println!("{:?}", dLs);
        let hiddens: Vec<_> = self
            .hidden_layer
            .iter()
            .map(|hidden| hidden.feed_forward(inputs))
            .collect();
        let doutputs: Vec<_> = self
            .outputs
            .iter()
            .map(|output| output.back_propogate(hiddens.as_slice()))
            .collect();
        let dhidden: Vec<_> = self
            .hidden_layer
            .iter()
            .map(|hidden| hidden.back_propogate(inputs))
            .collect();

        let dinfluence: Vec<_> = self
            .outputs
            .iter()
            .flat_map(|output| output.influence(hiddens.as_slice()))
            .collect();

        let new_outputs: Vec<_> = dLs
            .iter()
            .zip(doutputs.iter())
            .zip(self.outputs.iter())
            .map(|((dL, doutput), output)| {
                neuron::OutputNeuron::new(
                    doutput
                        .get_weights()
                        .iter()
                        .zip(output.get_weights().iter())
                        .map(|(dweight, weight)| weight - 0.1 * dL * dweight)
                        .collect(),
                    output.get_bias() - doutput.get_bias() * dL * 0.1,
                )
            })
            .collect();
        let new_hidden_layer: Vec<_> = dinfluence
            .iter()
            .zip(dhidden.iter())
            .zip(self.hidden_layer.iter())
            .map(|((influence, dhidden), hidden)| {
                neuron::HiddenNeuron::new(
                    hidden
                        .get_weights()
                        .iter()
                        .zip(dhidden.get_weights().iter())
                        .map(|(weight, dweight)| {
                            weight - 0.1 * dLs.first().unwrap() * dweight * influence
                        })
                        .collect(),
                    hidden.get_bias() - 0.1 * dLs.first().unwrap() * dhidden.get_bias() * influence,
                )
            })
            .collect();
        return Network {
            hidden_layer: new_hidden_layer,
            outputs: new_outputs,
        };
    }

    pub fn train(&mut self, data: &[Vec<f64>], actuals: &[Vec<f64>], epochs: usize) {
        for epoch in 0..epochs {
            for (entity, actual) in data.iter().zip(actuals.iter()) {
                let network = self.back_propogate(entity.as_slice(), actual);
                self.hidden_layer = network.hidden_layer;
                self.outputs = network.outputs;
                println!("{} h1 {:?}", epoch, self.hidden_layer.first().unwrap());
            }
        }
    }

    fn feed_forward_neurons<T>(&self, neurons: &Vec<T>, inputs: &[f64]) -> Vec<f64>
    where
        T: neuron::Neuron,
    {
        return neurons
            .as_slice()
            .iter()
            .map(|neuron| neuron.feed_forward(inputs))
            .collect();
    }
}

#[test]
fn feed_forward() {
    let weights = vec![0.0, 1.0];
    let bias = 0.0;
    let inputs = [2.0, 3.0];

    let network = Network {
        hidden_layer: vec![
            neuron::HiddenNeuron::new(weights.clone(), bias),
            neuron::HiddenNeuron::new(weights.clone(), bias),
        ],
        outputs: vec![neuron::OutputNeuron::new(weights, bias)],
    };

    assert_eq!(vec![0.7216325609518421], network.feed_forward(&inputs))
}

#[test]
fn feed_forward_two_outputs() {
    let inputs = [1.0, 0.0];

    let network = Network {
        hidden_layer: vec![neuron::HiddenNeuron::new(
            vec![0.13436424411240122, 0.8474337369372327],
            0.763774618976614,
        )],
        outputs: vec![
            neuron::OutputNeuron::new(vec![0.2550690257394217], 0.49543508709194095),
            neuron::OutputNeuron::new(vec![0.4494910647887381], 0.651592972722763),
        ],
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
            neuron::HiddenNeuron::new(vec![1.0, 1.0], 0.0),
            neuron::HiddenNeuron::new(vec![1.0, 1.0], 0.0),
        ],
        outputs: vec![neuron::OutputNeuron::new(vec![1.0, 1.0], 0.0)],
    };

    assert_eq!(vec![0.5236951740839997], network.feed_forward(&inputs));
}

#[test]
fn back_propogate() {
    let inputs = [-2.0, -1.0];
    let actuals = [1.0];

    let network = Network {
        hidden_layer: vec![
            neuron::HiddenNeuron::new(vec![1.0, 1.0], 0.0),
            neuron::HiddenNeuron::new(vec![1.0, 1.0], 0.0),
        ],
        outputs: vec![neuron::OutputNeuron::new(vec![1.0, 1.0], 0.0)],
    };

    assert_eq!(
        Network {
            hidden_layer: vec![
                neuron::HiddenNeuron::new(
                    vec![0.9978530464734189, 0.9989265232367095],
                    0.0010734767632905556
                ),
                neuron::HiddenNeuron::new(
                    vec![0.9978530464734189, 0.9989265232367095],
                    0.0010734767632905556
                ),
            ],
            outputs: vec![neuron::OutputNeuron::new(
                vec![1.0011269220242958, 1.0011269220242958],
                0.02376175595284281
            )],
        },
        network.back_propogate(&inputs, &actuals)
    );
}

#[test]
fn train() {
    let mut network = Network::new(2, 2, 1);

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

    network.train(&dataset, &actuals, 2);
    assert_eq!(vec![0.5028947910075867], network.feed_forward(&[-7.0, -3.0]));
    assert_eq!(vec![0.8742266927103481], network.feed_forward(&[20.0, 2.0]));
}
