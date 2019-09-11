use super::math;

#[derive(Debug, PartialEq)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn feed_forward(&self, inputs: &[f64]) -> f64 {
        let dot_product = math::dot_product(inputs, self.weights.as_slice());
        return math::sigmoid(dot_product + self.bias);
    }

    pub fn back_propogate(&self, preds: &[f64]) -> Neuron {
        let sum = math::dot_product(preds, self.weights.as_slice());
        let dsum = math::deriv_sigmoid(sum);
        let weights: Vec<_> = preds.iter().map(|w| w * dsum).collect();
        let neuron = Neuron { weights, bias: dsum };
        println!("sum: {}, inputs: {:?}, neuron: {:?}", sum, preds, neuron);
        return neuron;
    }
}

#[test]
fn feed_forward() {
    let weights = vec![0.0, 1.0];
    let bias = 4.0;
    let inputs = [2.0, 3.0];

    let neuron = Neuron {
        weights,
        bias,
    };

    assert_eq!(0.9990889488055994, neuron.feed_forward(&inputs))
}
