# cuta - cuTensorAuxiliary

An auxiliary library for tensor operations on NVIDIA GPU.

## Functionalities
- `cuta::reshape` (slow :( )

### cuTENSOR support
- `cuta::mode_t` to `cutensorTensorDescriptor_t` converter
- `cuta::mode_t` to mode name list (`std::vector<int32_t>`) converter

### Utils
- `cuta::utils::insert_mode(mode, "name", dim)` 
  - Insert `mode` to a mode named `name`.
- `cuta::utils::get_index(mode, pos)`
  - Get the memory index of an element in `mode` tensor at `pos`.
- `cuta::utils::get_num_elements(mode)`
  - Get the total number of elements in `mode` tensor
- `cuta::utils::get_permutation<T=unsigned>(mode, order)`
  - Get the permutation order array (e.g. `[1, 0, 2, 3]`)
- `cuta::utils::get_dim_sizes<T=unsigned>(mode)`
  - Extract dim sizes from `mode`


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
