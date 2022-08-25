use serde::Deserialize;
use serde_pickle::DeOptions;
use serde_pickle::Deserializer;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct PickleReader<R: Read + Sized> {
    de: Deserializer<R>,
}

impl<R: Read + Sized> PickleReader<R> {
    pub fn new(reader: R) -> PickleReader<R> {
        PickleReader {
            de: Deserializer::new(reader, DeOptions::new()),
        }
    }

    pub fn read_object<'de, T: Deserialize<'de>>(
        &mut self,
    ) -> Result<T, serde_pickle::error::Error> {
        self.de.reset_memo();
        let value = Deserialize::deserialize(&mut self.de)?;
        Ok(value)
    }
}

impl PickleReader<File> {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        Ok(PickleReader {
            de: Deserializer::new(file, DeOptions::new()),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pkl_reader() {
        let input = vec![
            0x80, 0x04, 0x95, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8c, 0x05, 0x48,
            0x65, 0x6c, 0x6c, 0x6f, 0x94, 0x2e, 0x80, 0x04, 0x95, 0x0a, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x8c, 0x06, 0x70, 0x69, 0x63, 0x6b, 0x6c, 0x65, 0x94, 0x2e, 0x00,
        ];
        let mut reader = PickleReader::new(std::io::Cursor::new(input.as_slice()));
        let string1: String = reader.read_object().unwrap();
        let string2: String = reader.read_object().unwrap();
        assert_eq!(&string1, "Hello");
        assert_eq!(&string2, "pickle");
    }
}
