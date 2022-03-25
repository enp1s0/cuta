# cuta - cuTensorAuxiliary

An auxiliary library for tensor operations on NVIDIA GPU.

## Functionalities
- `cuta::reshape` (slow :( )

### cuTENSOR support
- `cuta::mode_t` to `cutensorTensorDescriptor_t` converter
- `cuta::mode_t` to mode name list (`std::vector<int32_t>`) converter

## Installation
```bash
git clone https://github.com/enp1s0/cuta
cd cuta
mkdir build
cd build
cmake .. # Optional: -DCMALE_INSTALL_PREFIX=/path/to/install
make -j4
# Optional: make install
```

## License
MIT
