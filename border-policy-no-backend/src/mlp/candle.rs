use crate::Mat;
use candle_nn::VarMap;

impl super::Mlp {
    pub fn from_varmap(vm: &VarMap, w_names: &[&str], b_names: &[&str]) -> Self {
        let vars = vm.data().lock().unwrap();
        let ws: Vec<Mat> = w_names
            .iter()
            .map(|name| vars.get(*name).unwrap().as_tensor().clone().into())
            .collect();
        let bs: Vec<Mat> = b_names
            .iter()
            .map(|name| vars.get(*name).unwrap().as_tensor().clone().into())
            .collect();
        Self { ws, bs }
    }
}
