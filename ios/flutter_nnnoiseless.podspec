Pod::Spec.new do |s|
  s.name             = 'flutter_nnnoiseless'
  s.version          = '1.0.2'
  s.summary          = 'Rust-based audio noise reduction for Flutter.'
  s.description      = 'Flutter plugin backed by Rust for denoising audio.'
  s.homepage         = 'https://github.com/sk3llo/flutter_nnnoiseless'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Anton Karpenko' => 'kapraton@gmail.com' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency       'Flutter'
  s.platform         = :ios, '11.0'
  s.swift_version    = '5.0'

  s.vendored_frameworks = 'Frameworks/rust_lib_flutter_nnnoiseless.xcframework'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'LD_RUNPATH_SEARCH_PATHS' => '$(inherited) @executable_path/Frameworks @loader_path/Frameworks'
  }
end
