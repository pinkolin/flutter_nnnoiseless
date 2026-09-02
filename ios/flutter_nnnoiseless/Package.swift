// swift-tools-version: 5.9
// Swift Package Manager manifest for the iOS side of flutter_nnnoiseless.
//
// The Rust library is shipped as a prebuilt dynamic xcframework
// (scripts/build_apple_xcframework.sh); this package just vendors it. The
// plugin is an FFI plugin (no Dart-facing native class), so the product is the
// binary target alone. The podspec keeps CocoaPods working with the same
// framework path.
import PackageDescription

let package = Package(
    name: "flutter_nnnoiseless",
    platforms: [
        .iOS("13.0"),
    ],
    products: [
        .library(name: "flutter-nnnoiseless", targets: ["rust_lib_flutter_nnnoiseless"]),
    ],
    targets: [
        .binaryTarget(
            name: "rust_lib_flutter_nnnoiseless",
            path: "Frameworks/rust_lib_flutter_nnnoiseless.xcframework"
        ),
    ]
)
