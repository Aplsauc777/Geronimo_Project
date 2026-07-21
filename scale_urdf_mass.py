from pathlib import Path
import xml.etree.ElementTree as ET

INPUT_URDF = Path(
    r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\hexapod_robot_v3\urdf\geronimo_v3_collision.urdf"
)

OUTPUT_URDF = INPUT_URDF.with_name("geronimo_v3_collision_heavy.urdf")

TARGET_TOTAL_MASS_KG = 6.0

def main():
    tree = ET.parse(INPUT_URDF)
    root = tree.getroot()

    mass_tags = root.findall(".//mass")

    if not mass_tags:
        raise RuntimeError("No <mass> tags found in URDF.")
    
    current_total_mass = 0.0

    for mass_tag in mass_tags:
        current_total_mass += float(mass_tag.attrib["value"])

    scale_factor = TARGET_TOTAL_MASS_KG / current_total_mass

    print(f"Input URDF:  {INPUT_URDF}")
    print(f"Output URDF: {OUTPUT_URDF}")
    print(f"Current total mass: {current_total_mass:.6f} kg")
    print(f"Target total mass:  {TARGET_TOTAL_MASS_KG:.6f} kg")
    print(f"Scale factor:       {scale_factor:.3f}x")

    for mass_tag in mass_tags:
        old_mass = float(mass_tag.attrib["value"])
        new_mass = old_mass * scale_factor
        mass_tag.set("value", f"{new_mass:.9g}")

    inertia_attrs = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]

    for inertia_tag in root.findall(".//inertia"):
        for attr in inertia_attrs:
            if attr in inertia_tag.attrib:
                old_value = float(inertia_tag.attrib[attr])
                new_value = old_value * scale_factor
                inertia_tag.set(attr, f"{new_value:.9g}")

    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)

    print("\nDone.")
    print("Created heavier URDF:")
    print(OUTPUT_URDF)

if __name__ == "__main__":
    main()