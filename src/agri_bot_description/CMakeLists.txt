cmake_minimum_required(VERSION 3.8)
project(agri_bot_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  launch
  worlds
  param
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
