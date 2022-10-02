use ndarray::{arr1, Array1};

#[allow(dead_code)]
fn single(x: Array1<f32>, w: Array1<f32>, b: f32) -> f32 {
    let t = b + x.dot(&w);
    if t < 0.0 {
        return 0.0;
    } else {
        return 1.0;
    }
}

#[allow(dead_code)]
fn and(x1: f32, x2: f32) -> f32 {
    let x = arr1(&[x1, x2]);
    let w = arr1(&[0.5, 0.5]);
    let b = -0.7;

    return single(x, w, b);
}

#[allow(dead_code)]
fn nand(x1: f32, x2: f32) -> f32 {
    let x = arr1(&[x1, x2]);
    let w = arr1(&[-0.5, -0.5]);
    let b = 0.7;

    return single(x, w, b);
}

#[allow(dead_code)]
fn or(x1: f32, x2: f32) -> f32 {
    let x = arr1(&[x1, x2]);
    let w = arr1(&[0.5, 0.5]);
    let b = -0.2;

    return single(x, w, b);
}

#[allow(dead_code)]
fn xor(x1: f32, x2: f32) -> f32 {
    let s1 = nand(x1, x2);
    let s2 = or(x1, x2);
    return and(s1, s2);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_perceptron() {
        assert!(and(0.0, 0.0) < 0.5);
        assert!(and(0.0, 1.0) < 0.5);
        assert!(and(1.0, 0.0) < 0.5);
        assert!(and(1.0, 1.0) > 0.5);

        assert!(nand(0.0, 0.0) > 0.5);
        assert!(nand(0.0, 1.0) > 0.5);
        assert!(nand(1.0, 0.0) > 0.5);
        assert!(nand(1.0, 1.0) < 0.5);

        assert!(or(0.0, 0.0) < 0.5);
        assert!(or(0.0, 1.0) > 0.5);
        assert!(or(1.0, 0.0) > 0.5);
        assert!(or(1.0, 1.0) > 0.5);

        assert!(xor(0.0, 0.0) < 0.5);
        assert!(xor(0.0, 1.0) > 0.5);
        assert!(xor(1.0, 0.0) > 0.5);
        assert!(xor(1.0, 1.0) < 0.5);
    }
}
