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
