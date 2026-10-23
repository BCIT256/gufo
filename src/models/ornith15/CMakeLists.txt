# Ornith-1.5-35B-A3B (qwen35moe) model package. Configuration and weight
# binding only; execution routes are added once each is qualified.
add_library(gufo_ornith15 STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_ornith15 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ornith15 PUBLIC gufo_core)
