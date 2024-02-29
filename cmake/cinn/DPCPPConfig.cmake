#########################
#  FindDPCPP.cmake
#########################

include_guard()

include(FindPackageHandleStandardArgs)

# set DPCPP_ROOT
if(IS_DIRECTORY ${CINN_WITH_SYCL})
    set(DPCPP_ROOT ${CINN_WITH_SYCL})
else()
    EXECUTE_PROCESS(COMMAND which sycl-ls
                TIMEOUT 2
                OUTPUT_VARIABLE sycl_ls_path
                OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS ${sycl_ls_path})
        get_filename_component(DPCPP_ROOT ${sycl_ls_path}/../.. ABSOLUTE)
    else()
        message(FATAL_ERROR "Failed in SYCL path auto search, please set CINN_WITH_SYCL to sycl root path.")
    endif()
endif()
# find libsycl.so
find_library(DPCPP_LIB NAMES sycl PATHS "${DPCPP_ROOT}/lib")
find_package_handle_standard_args(DPCPP
    FOUND_VAR     DPCPP_FOUND
    REQUIRED_VARS DPCPP_LIB
)
if(NOT DPCPP_FOUND)
    return()
endif()
message(STATUS "Enable SYCL: " ${DPCPP_ROOT})
include_directories("${DPCPP_ROOT}/include/sycl;${DPCPP_ROOT}/include")
link_libraries(${DPCPP_LIB})
add_definitions(-DSYCL_CXX_COMPILER="${DPCPP_ROOT}/bin/clang++")

# set ONEDNN_ROOT
if(IS_DIRECTORY ${CINN_WITH_ONEDNN})
    set(ONEDNN_ROOT ${CINN_WITH_ONEDNN})
else()
    set(ONEDNN_ROOT $ENV{ONEDNN_ROOT})
endif()

# find libdnnl.so
find_library(ONEDNN_LIB NAMES sycl PATHS "${ONEDNN_ROOT}/build/src")
find_package_handle_standard_args(ONEDNN
    FOUND_VAR     ONEDNN_FOUND
    REQUIRED_VARS ONEDNN_LIB
)
if(NOT DPCPP_FOUND)
    return()
endif()
message(STATUS "Enable ONEDNN: " ${ONEDNN_ROOT})
include_directories("${ONEDNN_ROOT}/include")
link_libraries(${ONEDNN_LIB})