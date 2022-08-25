
use mnist::*;
use ndarray::prelude::*;

#[allow(dead_code)]
fn load_mnist(normalize : bool) -> (Array3<f32>, Array2<f32>, Array3<f32>, Array2<f32>){
   // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    
    let (train_data, test_data) = if normalize {
        (
            Array3::from_shape_vec((50_000, 28, 28), trn_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32 / 256.0),
            Array3::from_shape_vec((10_000, 28, 28), tst_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32 / 256.0)
        )
    } else {
        (
            Array3::from_shape_vec((50_000, 28, 28), trn_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32),
            Array3::from_shape_vec((10_000, 28, 28), tst_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f32)
        )
    };
   
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);   
    
    (train_data, train_labels, test_data, test_labels)
}

fn flatten(data : Array3<f32>) -> Array2<f32> {
    let shape = data.shape();
    let n = shape[0];
    let m = shape.iter().product::<usize>() / n;
    data.into_shape((n, m)).unwrap()
}

fn one_hot_encoded(data : Array2<f32>, n : usize) -> Array2<u8>{
    let shape = data.shape();
    if shape[1] != 1{
        panic!("Data should has a form of (n, 1)");
    }
    let mut res = Array2::zeros((shape[0], n));
    for (idx, v) in data.iter().enumerate(){
        let u = *v as usize;
        if u >= n {
            panic!("Data value should be smaller than {}", n);
        }

        res[[idx, u]] = 1;
    }
    return res;
}