# Synthetic-file checks run everywhere; the artifact check needs the pinned
# APEX MTP GGUF and reports a skip, not a pass, without it.
add_custom_target(ornith15_tests)

function(add_ornith15_test name labels)
  set(target ornith15_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE gufo_ornith15 ${ARGN})
  add_test(NAME ornith15.${name} COMMAND ${target})
  set_tests_properties(ornith15.${name} PROPERTIES
    LABELS "ornith15;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(ornith15_tests ${target})
endfunction()

add_ornith15_test(config "cpu")
add_ornith15_test(weights "cpu")
add_ornith15_test(artifact "cpu;external-model")
add_ornith15_test(chat_template "cpu")
add_ornith15_test(chat_tokens "cpu;external-model;tokenization")
foreach(name IN ITEMS chat_template chat_tokens)
  target_compile_definitions(ornith15_${name}_test PRIVATE
    GUFO_ORNITH15_CHAT_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/ornith15_chat_goldens.json"
    GUFO_ORNITH15_CHAT_TEMPLATE="${PROJECT_SOURCE_DIR}/tests/fixtures/ornith15_chat_template.jinja")
endforeach()
# Runs the scalar reference over the captured llama.cpp prompts.
add_ornith15_test(reference_oracle "cpu;external-model;correctness"
  gufo_ornith15_reference)
set_tests_properties(ornith15.reference_oracle PROPERTIES TIMEOUT 3600)

if (ENGINE_ENABLE_HIP)
  add_ornith15_test(gpu_ops "gpu;hip;gfx1151"
    gufo_ornith15_gpu gufo_ornith15_reference)
  add_ornith15_test(representation_gpu "gpu;hip;gfx1151"
    gufo_ornith15_gpu)
  add_ornith15_test(layer0_replay "gpu;hip;gfx1151;external-model"
    gufo_ornith15_gpu gufo_ornith15_reference)
  target_compile_definitions(ornith15_layer0_replay_test PRIVATE
    GUFO_ORNITH15_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  set_tests_properties(ornith15.layer0_replay PROPERTIES
    TIMEOUT 3600 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  add_test(NAME ornith15.layer0_exit_logic COMMAND ornith15_layer0_replay_test)
  set_tests_properties(ornith15.layer0_exit_logic PROPERTIES
    ENVIRONMENT "GUFO_ORNITH15_LAYER0_MODE=prove-exit"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LABELS "ornith15;gpu;hip" SKIP_RETURN_CODE 77)
  add_ornith15_test(gpu_decode "gpu;hip;gfx1151;external-model"
    gufo_ornith15_gpu gufo_ornith15_reference)
  set_tests_properties(ornith15.gpu_decode PROPERTIES TIMEOUT 3600)
  add_ornith15_test(b_prefix "gpu;hip;gfx1151;external-model"
    gufo_ornith15_gpu gufo_ornith15_reference)
  set_tests_properties(ornith15.b_prefix PROPERTIES
    TIMEOUT 3600 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  add_ornith15_test(qkv_downstream "gpu;hip;gfx1151;external-model"
    gufo_ornith15_gpu_probe gufo_ornith15_reference)
  target_compile_definitions(ornith15_qkv_downstream_test PRIVATE
    GUFO_ORNITH15_TEST_PROBE)
  set_tests_properties(ornith15.qkv_downstream PROPERTIES
    TIMEOUT 3600 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  add_ornith15_test(route_audit "gpu;hip;gfx1151;external-model"
    gufo_ornith15_gpu_probe gufo_ornith15_reference)
  target_compile_definitions(ornith15_route_audit_test PRIVATE
    GUFO_ORNITH15_TEST_PROBE)
  set_tests_properties(ornith15.route_audit PROPERTIES
    TIMEOUT 3600 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()
