use rand::{self, Rng};
use universal_sampler::FunctionSampler;

fn main() {
    let rng = rand::thread_rng();

    let sampler = FunctionSampler::new(|x| -x * x + x, 0.0..1.0, 30).unwrap();

    //println!("{:#?}", sampler.hist);

    for s in rng.sample_iter(&sampler).take(100) {
        println!("{s}")
    }
}
