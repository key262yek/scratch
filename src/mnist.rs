use mnist::*;
use ndarray::prelude::*;

#[allow(dead_code)]
fn load_mnist(
    normalize: bool,
    one_hot_encoded: bool,
) -> (Array3<f32>, Array2<f32>, Array3<f32>, Array2<f32>) {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = if one_hot_encoded {
        MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .label_format_one_hot()
            .finalize()
    } else {
        MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize()
    };
    let (train_data, test_data) = if normalize {
        (
            Array3::from_shape_vec((50_000, 28, 28), trn_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32 / 256.0),
            Array3::from_shape_vec((10_000, 28, 28), tst_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32 / 256.0),
        )
    } else {
        (
            Array3::from_shape_vec((50_000, 28, 28), trn_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32),
            Array3::from_shape_vec((10_000, 28, 28), tst_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32),
        )
    };

    let n = if one_hot_encoded { 10 } else { 1 };
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, n), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, n), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, test_data, test_labels)
}

#[allow(dead_code)]
fn flatten(data: Array3<f32>) -> Array2<f32> {
    let shape = data.shape();
    let n = shape[0];
    let m = shape.iter().product::<usize>() / n;
    data.into_shape((n, m)).unwrap()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore]
    fn test_read() {
        let dataset = load_mnist(false, false);
        assert_eq!(dataset.0.shape(), [50000, 28, 28]);

        let flat = flatten(dataset.0);
        assert_eq!(flat.shape(), [50000, 784]);
    }
}
