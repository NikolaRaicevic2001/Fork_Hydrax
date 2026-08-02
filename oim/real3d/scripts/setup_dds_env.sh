#!/usr/bin/env bash
# Source this (`. setup_dds_env.sh`) in EVERY ROS terminal on BOTH machines
# (perception laptop running FoundationPose, and the desktop running the
# planner) so their ROS 2 topics + TF talk to each other over the LAN.
#
# This is the recurring cross-machine setup from OI-MPPI. Get all three right
# or discovery silently fails (nodes see nothing from the other host):
#   1. same RMW (CycloneDDS),
#   2. same ROS_DOMAIN_ID,
#   3. a CYCLONEDDS_URI whose <Peer> is the other host and whose interface is
#      this host's LAN interface (edit config/cyclonedds.xml first).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}     # set the SAME number on both machines
export CYCLONEDDS_URI="file://${HERE}/../config/cyclonedds.xml"

echo "[dds] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "[dds] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[dds] CYCLONEDDS_URI=$CYCLONEDDS_URI"
echo "[dds] verify cross-machine: 'ros2 topic list' should show the other host's"
echo "      topics (e.g. /object_pose, /joint_states); 'ros2 topic echo /tf' should"
echo "      carry fp_object_pose. If empty, check the interface name + Peer IP in"
echo "      config/cyclonedds.xml and that both hosts are on the same subnet."
