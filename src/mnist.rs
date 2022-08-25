use std::path::Path;

const URL_BASE: &str = "http://yann.lecun.com/exdb/mnist/";
const KEY_FILE: [&str; 4] = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

const DS_DIR: &str = "./dataset/mnist";
const SAVE_DIR: &str = "./dataset/mnist/mnist.pkl";

const NUM_TRAIN: usize = 60000;
const NUM_TEST: usize = 10000;
const DIM_IMG: (usize, usize, usize) = (1, 28, 28);
const SIZE_IMG: usize = 784;

pub fn download(name: &str) {
    let mut path = DS_DIR.to_string();
    path.push_str("/");
    path.push_str(name);

    if Path::new(path).exists() {
        return;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_download() {
        download("");
    }
}
