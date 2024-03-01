
# 提取当前目录及其子目录下所有文件，形成源文件列表

# 设置头文件路径
include_directories(common)
include_directories(src)

file(GLOB_RECURSE SRC_LIST "*.cpp" "*.h" "*.cc" "*.hpp" )
# aux_source_directory(. SRC_LIST)

# 打印遍历SRC_LIST
foreach(f ${SRC_LIST})
    message(STATUS "SRC_LIST: ${f}")
endforeach(f)

# 设置库生成路径
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/libs)

# # 生成zj_video_pipeline库
# add_library(logger SHARED  ${CMAKE_SOURCE_DIR}/src/logger/easylogging++.cc)

add_library(zj_pipe SHARED ${SRC_LIST})

