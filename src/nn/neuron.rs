use super::math;

#[derive(Debug, PartialEq)]
pub struct NeuronProperties {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[derive(Debug, PartialEq)]
pub enum Neuron {
    Output(NeuronProperties),
    Hidden(NeuronProperties)
}

impl Neuron {
    pub fn feed_forward(&self, inputs: &[f64]) -> f64 {
        match self {
            Neuron::Output(neuron) |
            Neuron::Hidden(neuron) => {
                let dot_product = math::dot_product(inputs, neuron.weights.as_slice());
                return math::sigmoid(dot_product + neuron.bias);
            }
        }
    }

    pub fn back_propogate(&self, preds: &[f64]) -> Neuron {
        match self {
            Neuron::Output(neuron) => {
                let sum = math::dot_product(preds, neuron.weights.as_slice());
                let dsum = math::deriv_sigmoid(sum);
                let weights: Vec<_> = preds.iter().map(|w| w * dsum).collect();
                let new_neuron = Neuron::Output(NeuronProperties { weights, bias: dsum });
                println!("sum: {}, inputs: {:?}, neuron: {:?}", sum, preds, new_neuron);
                return new_neuron;
            }
            Neuron::Hidden(neuron) => {
                let sum = math::dot_product(preds, neuron.weights.as_slice());
                let dsum = math::deriv_sigmoid(sum);
                let weights: Vec<_> = preds.iter().map(|w| w * dsum).collect();
                let new_neuron = Neuron::Hidden(NeuronProperties { weights, bias: dsum });
                println!("sum: {}, inputs: {:?}, neuron: {:?}", sum, preds, new_neuron);
                return new_neuron;
            }
        }
    }
}

#[test]
fn feed_forward() {
    let weights = vec![0.0, 1.0];
    let bias = 4.0;
    let inputs = [2.0, 3.0];

    let neuron = Neuron::Output(NeuronProperties {
        weights,
        bias,
    });

    assert_eq!(0.9990889488055994, neuron.feed_forward(&inputs))
}
