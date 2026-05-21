cmake_minimum_required(VERSION 3.8)
project(agri_bot_navigation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  param
  maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
