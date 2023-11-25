use std::f64::consts::E;

pub enum Activation {
	IDENTITY,
	SIGMOID,
	TANH,
	RELU,
}

impl Activation {
	pub fn function(&self, x: f64) -> f64 {
		match &self {
			Activation::IDENTITY => x,
			Activation::SIGMOID => 1.0 / (1.0 + E.powf(-x)),
			Activation::TANH => x.tanh(),
			Activation::RELU => x.max(0.0),
		}
	}
	pub fn derivative(&self, x: f64) -> f64 {
		match &self {
			Activation::IDENTITY => 1.0,
			Activation::SIGMOID => x * (1.0 - x),
			Activation::TANH => 1.0 - (x.powi(2)),
			Activation::RELU => {
				if x > 0.0 {
					1.0
				} else {
					0.0
				}
			}
		}
	}
}
