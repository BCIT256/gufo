# Ornith-1.5-35B-A3B (qwen35moe) model package: configuration, weight
# binding and chat policy. Execution routes are added once each is qualified.
add_library(gufo_ornith15 STATIC
  chat_template.cpp
  config.cpp
  weights.cpp)

target_include_directories(gufo_ornith15 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ornith15 PUBLIC gufo_core)

# Scalar float32 oracle, built only by the validation targets that link it.
add_library(gufo_ornith15_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_ornith15_reference PUBLIC gufo_ornith15)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  add_library(gufo_ornith15_gpu STATIC
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp
    kernels/rocm/kernels.hip.cpp)
  set_source_files_properties(kernels/rocm/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_ornith15_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>)
  target_link_libraries(gufo_ornith15_gpu PUBLIC
    gufo_ornith15 gufo_qwen38_flash_next gufo_qwen38_flash_next_mmq hip::host)

  # Test-only session. The production archive is built without
  # GUFO_ORNITH15_TEST_PROBE, so its Step cannot replace QKV.
  add_library(gufo_ornith15_gpu_probe STATIC EXCLUDE_FROM_ALL
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp
    kernels/rocm/kernels.hip.cpp)
  set_source_files_properties(kernels/rocm/kernels.hip.cpp
    TARGET_DIRECTORY gufo_ornith15_gpu_probe PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gufo_ornith15_gpu_probe PRIVATE
    GUFO_ORNITH15_TEST_PROBE)
  target_compile_options(gufo_ornith15_gpu_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>)
  target_link_libraries(gufo_ornith15_gpu_probe PUBLIC
    gufo_ornith15 gufo_qwen38_flash_next gufo_qwen38_flash_next_mmq hip::host)
endif()
