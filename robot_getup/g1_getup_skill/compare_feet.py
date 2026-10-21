import sys, xml.etree.ElementTree as ET, numpy as np, mujoco
URDF, MJ1, MJ2 = sys.argv[1:4]
print("=== Isaac 학습 URDF")
root = ET.parse(URDF).getroot()
for ln in ("left_ankle_roll_link", "left_wrist_yaw_link"):
    link = root.find(f"link[@name='{ln}']")
    for c in link.findall("collision"):
        o = c.find("origin"); g = list(c.find("geometry"))[0]
        print(f"  {ln:22s} {g.tag:8s} {g.attrib}  위치 {o.attrib.get('xyz') if o is not None else '0 0 0'}")
for path in (MJ1, MJ2):
    print("=== MuJoCo", path.split("/")[-1])
    m = mujoco.MjModel.from_xml_path(path)
    for ln in ("left_ankle_roll_link", "left_wrist_yaw_link"):
        b = m.body(ln).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                print(f"  {ln:22s} {mujoco.mjtGeom(m.geom_type[g]).name[7:]:8s} size {np.round(m.geom_size[g],3)}  위치 {np.round(m.geom_pos[g],3)}")
