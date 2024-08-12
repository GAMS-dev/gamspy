cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SimpleExternalModule)


add_library(simple_ext_module SHARED mylib.cpp)

target_compile_definitions(simple_ext_module PRIVATE CONST_DERIV=1)
target_compile_definitions(simple_ext_module PRIVATE HV_PRODUCT=1)
target_include_directories(simple_ext_module PRIVATE ".")

set_property(TARGET simple_ext_module PROPERTY CXX_STANDARD 17)
set_property(TARGET simple_ext_module PROPERTY POSITION_INDEPENDENT_CODE ON)
