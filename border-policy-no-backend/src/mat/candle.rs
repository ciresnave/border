impl From<candle_core::Tensor> for super::Mat {
    fn from(x: candle_core::Tensor) -> Self {
        let shape: Vec<i32> = x.dims().iter().map(|e| *e as i32).collect();
        let data = match shape.len() {
            1 => x.to_vec1::<f32>().unwrap(),
            2 => x.to_vec2::<f32>().unwrap().into_iter().flatten().collect(),
            _ => panic!("Invalid matrix size: {:?}", shape),
        };
        let shape = match shape.len() {
            1 => vec![shape[0], 1],
            2 => shape,
            _ => panic!("Invalid matrix size: {:?}", shape),
        };
        Self { data, shape }
    }
}
