use crate::Mat;
use tch::nn::VarStore;

impl super::Mlp {
    pub fn from_varstore(vs: &VarStore, w_names: &[&str], b_names: &[&str]) -> Self {
        let vars = vs.variables();
        let ws: Vec<Mat> = w_names
            .iter()
            .map(|name| vars[&name.to_string()].copy().into())
            .collect();
        let bs: Vec<Mat> = b_names
            .iter()
            .map(|name| vars[&name.to_string()].copy().into())
            .collect();

        Self { ws, bs }
    }
}
