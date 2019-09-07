use super::math;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn feed_forward(&self, inputs: &[f64]) -> f64 {
        let dot_product: f64 = inputs
            .into_iter()
            .zip(self.weights.as_slice().into_iter())
            .map(|(x, y)| x * y)
            .sum();
        return math::sigmoid(dot_product + self.bias);
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
