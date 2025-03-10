# Try to find libraries for FFmpeg
# Once done this will define
#
#  FFMPEG_FOUND - system has FFmpeg
#  FFMPEG_INCLUDE_DIRS - the FFmpeg include directories
#  FFMPEG_LIBRARIES - link these to use FFmpeg

if(WIN32)
    set(FFMPEG_INCLUDE_PATHS "C:/ffmpeg/include")
    set(FFMPEG_LIBRARY_PATHS "C:/ffmpeg/lib")
elseif(UNIX AND NOT APPLE)
    set(FFMPEG_INCLUDE_PATHS "/usr/local/include" "/usr/include")
    set(FFMPEG_LIBRARY_PATHS "/usr/local/lib" "/usr/lib")
else()
    message(FATAL_ERROR "Unsupported platform for FFmpeg")
endif()

find_path(FFMPEG_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS /usr/local/include /usr/include
)

find_library(FFMPEG_LIBAVCODEC
    NAMES avcodec
    PATHS /usr/local/lib /usr/lib
)

find_library(FFMPEG_LIBAVFORMAT
    NAMES avformat
    PATHS /usr/local/lib /usr/lib
)

find_library(FFMPEG_LIBAVUTIL
    NAMES avutil
    PATHS /usr/local/lib /usr/lib
)

find_library(FFMPEG_LIBSWSCALE
    NAMES swscale
    PATHS /usr/local/lib /usr/lib
)

set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
set(FFMPEG_LIBRARIES ${FFMPEG_LIBAVCODEC} ${FFMPEG_LIBAVFORMAT} ${FFMPEG_LIBAVUTIL} ${FFMPEG_LIBSWSCALE})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg DEFAULT_MSG FFMPEG_INCLUDE_DIR FFMPEG_LIBRARIES)