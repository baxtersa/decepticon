pub fn dot_product(xs: &[f64], ys: &[f64]) -> f64 {
    return xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| x * y)
        .sum();
}

pub fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

pub fn deriv_sigmoid(x: f64) -> f64 {
    let fx = sigmoid(x);
    return fx * (1.0 - fx);
}

pub fn mse(expecteds: &[f64], actuals: &[f64]) -> f64 {
    let sum_squared_errors: f64 = expecteds
        .iter()
        .zip(actuals.iter())
        .map(|(x, y)| (x - y).powf(2.0))
        .sum();
    return sum_squared_errors / expecteds.len() as f64;
}

#[test]
fn test_mse() {
    let expecteds = [0.0, 0.0, 0.0, 0.0];
    let actuals = [1.0, 0.0, 0.0, 1.0];

    assert_eq!(0.5, mse(&expecteds, &actuals))
}
