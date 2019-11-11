mod nn;

use nn::network::Network;

fn main() {
    let mut network = Network::new(2, 2, 1);

    let dataset = [
        vec![-2.0, -1.0],
        vec![25.0, 6.0],
        vec![17.0, 4.0],
        vec![-15.0, -6.0],
    ];
    let actuals = [vec![1.0], vec![0.0], vec![0.0], vec![1.0]];

    network.train(&dataset, &actuals, 1000);

    let emily = [-7.0, -3.0]; // 128 pounds, 63 inches
    let frank = [20.0, 2.0];  // 155 pounds, 68 inches
    println!("Emily: {:?}", network.predict(&emily)); // 0.951 - F
    println!("Frank: {:?}", network.predict(&frank)); // 0.039 - M

}
