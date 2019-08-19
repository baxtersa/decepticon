use super::neuron;

struct Network<'a> {
    weights: &'a [f64],
    bias: f64,
    hidden_layer: [neuron::Neuron<'a>; 2],
    output: neuron::Neuron<'a>,
}

impl<'a> Network<'a> {
    fn new(weights: &'a [f64], bias: f64) -> Network<'a> {
        return Network {
            weights,
            bias,
            hidden_layer: [
                neuron::Neuron { weights, bias },
                neuron::Neuron { weights, bias },
            ],
            output: neuron::Neuron { weights, bias },
        };
    }

    fn feed_forward(&self, inputs: &[f64]) -> f64 {
        let hidden_outs: Vec<_> = self
            .hidden_layer
            .iter()
            .map(|neuron| neuron.feed_forward(inputs))
            .collect();
        return self.output.feed_forward(&hidden_outs);
    }
}

#[test]
fn feed_forward() {
    let weights = [0.0, 1.0];
    let bias = 0.0;
    let inputs = [2.0, 3.0];

    let network = Network::new(&weights, bias);

    assert_eq!(0.7216325609518421, network.feed_forward(&inputs))
}
