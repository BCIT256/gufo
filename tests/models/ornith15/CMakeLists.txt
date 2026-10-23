# Synthetic-file checks run everywhere; the artifact check needs the pinned
# APEX MTP GGUF and reports a skip, not a pass, without it.
add_custom_target(ornith15_tests)

function(add_ornith15_test name labels)
  set(target ornith15_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE gufo_ornith15)
  add_test(NAME ornith15.${name} COMMAND ${target})
  set_tests_properties(ornith15.${name} PROPERTIES
    LABELS "ornith15;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(ornith15_tests ${target})
endfunction()

add_ornith15_test(config "cpu")
add_ornith15_test(weights "cpu")
add_ornith15_test(artifact "cpu;external-model")
