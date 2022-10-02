use ndarray::Array1;

#[allow(dead_code)]
fn node(x: &Array1<f32>, w: &Array1<f32>, h: &dyn Fn(f32) -> f32) -> f32 {
    h(x.dot(w))
}

#[allow(dead_code)]
fn step_fn(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    } else {
        return 1.0;
    }
}

#[allow(dead_code)]
fn step_fn_arr(mut x: Array1<f32>) -> Array1<f32> {
    x.mapv_inplace(step_fn);
    x
}

#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + (-x).exp());
}

#[allow(dead_code)]
fn sigmoid_arr(mut x: Array1<f32>) -> Array1<f32> {
    x.mapv_inplace(sigmoid);
    x
}

#[allow(dead_code)]
fn relu(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    } else {
        return x;
    }
}

#[allow(dead_code)]
fn relu_arr(mut x: Array1<f32>) -> Array1<f32> {
    x.mapv_inplace(relu);
    x
}

#[allow(dead_code)]
fn softmax_arr(mut x: Array1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| if a < b { b } else { a });
    x.mapv_inplace(|s| (s - max).exp());

    let sum = x.sum();
    x.mapv_inplace(|s| s / sum);
    x
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        let test = sigmoid_arr(array![-1.0, 1.0, 2.0]);
        let ans = array![0.26894142, 0.73105858, 0.88079708];

        assert_abs_diff_eq!(test, ans, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax() {
        let test1 = softmax_arr(array![0.3, 2.9, 4.0]);
        let ans = array![0.01821127, 0.24519181, 0.73659691];

        assert_abs_diff_eq!(test1, ans, epsilon = 1e-5);

        let test2 = softmax_arr(array![1010f32, 1000f32, 990f32]);
        let ans = array![9.999546e-1, 4.53978686e-5, 2.06106005e-9];

        assert_abs_diff_eq!(test2, ans, epsilon = 1e-5);
    }
}
