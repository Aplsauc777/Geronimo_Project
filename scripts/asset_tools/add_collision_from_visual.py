# ------------------------------------------------------------
# add_collision_from_visual.py
#
# Purpose:
# Your Onshape URDF may have visual meshes but no collision meshes.
#
# In Isaac Sim / Isaac Lab:
# - visual geometry = what the robot looks like
# - collision geometry = what physics can touch
#
# If the robot has visuals but no collisions, it can appear on screen
# but fall straight through the ground.
#
# This script opens your URDF, checks every link, and if a link has
# visuals but no collisions, it copies the visual geometry and uses it
# as collision geometry.
# ------------------------------------------------------------
# Chungus

# copy lets us duplicate XML elements safely.
# We need this because we are copying <origin> and <geometry>
# from <visual> into a new <collision>.
import copy

# xml.etree.ElementTree lets Python read and edit XML files.
# URDF files are XML files.
import xml.etree.ElementTree as ET

# pathlib makes file paths cleaner and easier to work with.
from pathlib import Path


# ------------------------------------------------------------
# 1. Set the input and output URDF paths
# ------------------------------------------------------------

# This is your original URDF exported from Onshape.
input_urdf = Path(
    r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\hexapod_robot_v3\urdf\hexapod_robot_v3.urdf"
)

# This creates a new output file in the same folder.
# It does NOT overwrite the original URDF.
#
# Example:
# original: geronimo.urdf
# output:   geronimo_collision.urdf
output_urdf = input_urdf.with_name("geronimo_v3_collision.urdf")


# ------------------------------------------------------------
# 2. Load the URDF XML file
# ------------------------------------------------------------

# Parse/read the URDF file.
tree = ET.parse(input_urdf)

# Get the root XML element.
# In a URDF, this is usually the <robot> element.
root = tree.getroot()


# ------------------------------------------------------------
# 3. Track how many collision elements we add
# ------------------------------------------------------------

# This counter is just for printing a helpful result at the end.
added = 0


# ------------------------------------------------------------
# 4. Loop through every robot link
# ------------------------------------------------------------

# In URDF, each physical piece of the robot is usually a <link>.
# Example:
# <link name="body">
# <link name="front_left_femur">
# <link name="front_left_tibia">
for link in root.findall("link"):

    # Find all existing collision elements in this link.
    collisions = link.findall("collision")

    # Find all visual elements in this link.
    visuals = link.findall("visual")


    # --------------------------------------------------------
    # 5. Skip links that already have collision geometry
    # --------------------------------------------------------

    # If this link already has collision geometry, we do not touch it.
    # This avoids accidentally duplicating collisions.
    if collisions:
        continue


    # --------------------------------------------------------
    # 6. Copy each visual mesh into a collision mesh
    # --------------------------------------------------------

    # If the link has no collisions, we use its visual geometry
    # as its collision geometry.
    for visual in visuals:

        # Create a new <collision> XML element.
        collision = ET.Element("collision")

        # The <origin> element tells where the visual mesh is positioned
        # relative to the link.
        #
        # Example:
        # <origin xyz="0 0 0" rpy="0 0 0"/>
        origin = visual.find("origin")

        # The <geometry> element tells what mesh/shape the visual uses.
        #
        # Example:
        # <geometry>
        #   <mesh filename="meshes/body.stl"/>
        # </geometry>
        geometry = visual.find("geometry")

        # If the visual has an origin, copy it into the collision.
        # deepcopy is used so the new collision gets its own copy,
        # instead of reusing the original XML object.
        if origin is not None:
            collision.append(copy.deepcopy(origin))

        # If the visual has geometry, copy it into the collision.
        if geometry is not None:
            collision.append(copy.deepcopy(geometry))

        # Add the new <collision> element into the current <link>.
        link.append(collision)

        # Increase the counter so we know how many were added.
        added += 1


# ------------------------------------------------------------
# 7. Save the edited URDF
# ------------------------------------------------------------

# Write the modified XML tree to the new URDF file.
# encoding="utf-8" is standard.
# xml_declaration=True adds the first line:
# <?xml version='1.0' encoding='utf-8'?>
tree.write(output_urdf, encoding="utf-8", xml_declaration=True)


# ------------------------------------------------------------
# 8. Print the result
# ------------------------------------------------------------

# This tells you how many collision elements were created.
print(f"Added {added} collision elements.")

# This tells you where the new URDF was saved.
print(f"Wrote: {output_urdf}")