use super::math;

pub trait NeuronBase {
    fn new(weights: Vec<f64>, bias: f64) -> Self;
    fn get_weights(&self) -> &Vec<f64>;
    fn get_bias(&self) -> f64;
}

pub trait Neuron {
    fn feed_forward(&self, inputs: &[f64]) -> f64;
    fn back_propogate(&self, preds: &[f64]) -> Self;
}

impl<T: NeuronBase + std::fmt::Debug> Neuron for T {
    fn feed_forward(&self, inputs: &[f64]) -> f64 {
        let dot_product = math::dot_product(inputs, self.get_weights().as_slice());
        return math::sigmoid(dot_product + self.get_bias());
    }

    fn back_propogate(&self, preds: &[f64]) -> Self {
        let sum = math::dot_product(preds, self.get_weights().as_slice());
        let dsum = math::deriv_sigmoid(sum);
        let weights: Vec<_> = preds.iter().map(|w| w * dsum).collect();
        let new_neuron = Self::new(weights, dsum);
        return new_neuron;
    }
}

impl OutputNeuron {
    pub fn influence(&self, inputs: &[f64]) -> Vec<f64> {
        let sum = math::dot_product(inputs, self.get_weights().as_slice());
        let dsum = math::deriv_sigmoid(sum);
        let weights: Vec<_> = self.get_weights().iter().map(|w| w * dsum).collect();
        return weights;
    }
}

#[derive(Debug, PartialEq)]
pub struct HiddenNeuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[derive(Debug, PartialEq)]
pub struct OutputNeuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl NeuronBase for HiddenNeuron {
    fn new(weights: Vec<f64>, bias: f64) -> HiddenNeuron {
        return HiddenNeuron { weights, bias };
    }

    fn get_weights(&self) -> &Vec<f64> {
        return &self.weights;
    }

    fn get_bias(&self) -> f64 {
        return self.bias;
    }
}

impl NeuronBase for OutputNeuron {
    fn new(weights: Vec<f64>, bias: f64) -> OutputNeuron {
        return OutputNeuron { weights, bias };
    }

    fn get_weights(&self) -> &Vec<f64> {
        return &self.weights;
    }

    fn get_bias(&self) -> f64 {
        return self.bias;
    }
}

#[test]
fn feed_forward() {
    let weights = vec![0.0, 1.0];
    let bias = 4.0;
    let inputs = [2.0, 3.0];

    let neuron = OutputNeuron::new(weights, bias);
    assert_eq!(0.9990889488055994, neuron.feed_forward(&inputs))
}

#[test]
fn feed_forward_hidden() {
    let weights = vec![0.0, 1.0];
    let bias = 0.0;
    let inputs = [2.0, 3.0];

    let h1 = HiddenNeuron::new(weights.clone(), bias);

    let h1_out = h1.feed_forward(&inputs);
    assert_eq!(0.9525741268224334, h1_out);

    let output = OutputNeuron::new(weights, bias);
    assert_eq!(0.7216325609518421, output.feed_forward(&[h1_out, h1_out]));
}

#[test]
fn back_propogate_hidden() {
    let weights = vec![1.0, 1.0];
    let bias = 0.0;
    let inputs = [-2.0, -1.0];

    let h1 = HiddenNeuron::new(weights.clone(), bias);
    assert_eq!(
        HiddenNeuron::new(
            vec![-0.09035331946182427, -0.04517665973091214],
            0.04517665973091214
        ),
        h1.back_propogate(&inputs)
    );
}

#[test]
fn back_propogate_output() {
    let weights = vec![1.0, 1.0];
    let bias = 0.0;
    let h1 = HiddenNeuron::new(weights.clone(), bias);
    let h2 = HiddenNeuron::new(weights.clone(), bias);
    let inputs = [-2.0, -1.0];
    let h_inputs = [h1.feed_forward(&inputs), h2.feed_forward(&inputs)];

    let output = OutputNeuron::new(weights.clone(), bias);
    assert_eq!(
        OutputNeuron::new(
            vec![0.24943853872512894, 0.24943853872512894],
            0.24943853872512894
        ),
        output.back_propogate(&h_inputs)
    );
}
