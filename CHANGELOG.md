## 1.0.2
- Updated `flutter_rust_bridge` integration to 2.12.0
- Regenerated Rust/Dart bridge code for `flutter_rust_bridge` 2.12.0
- Fixed iOS runtime error when initializing the Rust library:
  `Failed to load dynamic library 'rust_lib_flutter_nnnoiseless.framework/rust_lib_flutter_nnnoiseless'`
  
## 1.0.1

- Changed method `denoiseInRealtime` to `denoiseChunk` for readability
- Added description to the `denoise` methods
- Removed rust_builder
- Updated README.md

## 1.0.0

* Initial release