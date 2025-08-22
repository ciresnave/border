impl From<tch::Tensor> for super::Mat {
    fn from(x: tch::Tensor) -> Self {
        let shape: Vec<i32> = x.size().iter().map(|e| *e as i32).collect();
        let (n, shape) = match shape.len() {
            1 => (shape[0] as usize, vec![shape[0], 1]),
            2 => ((shape[0] * shape[1]) as usize, shape),
            _ => panic!("Invalid matrix size: {:?}", shape),
        };
        let mut data: Vec<f32> = vec![0f32; n];
        x.f_copy_data(&mut data, n).unwrap();
        Self { data, shape }
    }
}
