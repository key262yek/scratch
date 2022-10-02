use ndarray::{Array1, Array2, Zip};

#[allow(dead_code)]
fn mse(expect: &Array1<f32>, real: &Array1<f32>) -> f32 {
    return 0.5 * (expect - real).mapv(|x| x.powi(2)).sum();
}

#[allow(dead_code)]
fn mse_batch(expect: &Array2<f32>, real: &Array2<f32>) -> f32 {
    let batch_size = expect.shape()[0] as f32;
    return 0.5 * (expect - real).mapv(|x| x.powi(2)).sum() / batch_size;
}

#[allow(dead_code)]
fn cee(expect: &Array1<f32>, real: &Array1<f32>) -> f32 {
    let delta = 1e-7;
    return -real.dot(&expect.mapv(|x| (x + delta).ln()));
}

#[allow(dead_code)]
fn cee_batch(expect: &Array2<f32>, real: &Array2<f32>) -> f32 {
    let batch_size = expect.shape()[0] as f32;
    let delta = 1e-7;
    return -real.dot(&expect.mapv(|x| (x + delta).ln()).t()).sum() / batch_size;
}

#[allow(dead_code)]
fn numerical_gradient(f: fn(&Array1<f32>) -> f32, x: &mut Array1<f32>, h: f32) -> Array1<f32> {
    let n = x.len();
    let mut grad = Array1::<f32>::zeros(n);

    for i in 0..n {
        x[i] += h;
        let f1 = f(x);

        x[i] -= 2. * h;
        let f2 = f(x);

        x[i] += h;
        grad[i] = (f1 - f2) / (2. * h);
        8
    }

    return grad;
}

#[allow(dead_code)]
fn gradient_descent(
    f: fn(&Array1<f32>) -> f32,
    mut x: Array1<f32>,
    lr: f32,
    n_step: usize,
) -> Array1<f32> {
    let h = 2f32.powi(-20);

    for _ in 0..n_step {
        let grad = numerical_gradient(f, &mut x, h);
        Zip::from(&mut x).and(&grad).for_each(|s, &g| *s -= g * lr);
    }

    return x;
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_mse() {
        let real = array![0., 0., 1., 0., 0., 0., 0., 0., 0., 0.];
        let expect1 = array![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
        assert_abs_diff_eq!(mse(&expect1, &real), 0.0975, epsilon = 1e-4);

        let expect2 = array![0.1, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(mse(&expect2, &real), 0.59625, epsilon = 1e-5);
    }

    #[test]
    fn test_cee() {
        let real = array![0., 0., 1., 0., 0., 0., 0., 0., 0., 0.];
        let expect1 = array![0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
        assert_abs_diff_eq!(cee(&expect1, &real), 0.5108, epsilon = 1e-4);

        let expect2 = array![0.1, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6, 0.0, 0.0, 0.0];
        assert_abs_diff_eq!(cee(&expect2, &real), 2.30258, epsilon = 1e-5);
    }

    fn func2(x: &Array1<f32>) -> f32 {
        x.map(|s| s * s).sum()
    }

    #[test]
    fn test_gradient() {
        let mut x = array![3.0, 4.0];
        assert_abs_diff_eq!(
            numerical_gradient(func2, &mut x, 2f32.powi(-20)),
            array![6., 8.],
            epsilon = 1e-2
        );

        x[0] = 0.0;
        x[1] = 2.0;
        assert_abs_diff_eq!(
            numerical_gradient(func2, &mut x, 2f32.powi(-20)),
            array![0., 4.],
            epsilon = 1e-2
        );

        x[0] = 3.0;
        x[1] = 0.0;
        assert_abs_diff_eq!(
            numerical_gradient(func2, &mut x, 2f32.powi(-20)),
            array![6., 0.],
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_gradient_descent() {
        let x = array![-3.0, 4.0];
        let min_x = gradient_descent(func2, x, 0.1, 100);

        assert_abs_diff_eq!(min_x, array![0.0, 0.0], epsilon = 1e-4);
    }
}
