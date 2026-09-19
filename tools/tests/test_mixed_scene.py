"""The mixed scene's geometry: frames, rotations, the shell's quads and the files' layout.

    python3 tools/tests/test_mixed_scene.py      (or pytest tools/tests)
"""

import json
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_scene as ms  # noqa: E402


def turn(axis, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def room(world=np.eye(3), metre=2.0):
    return SimpleNamespace(metre=metre, centre=np.array([10.0, 20.0]), world=world, half=np.array([4.0, 6.0]))


def test_quaternions_match_their_matrices():
    for axis, degrees in [((0, 0, 1), 30), ((1, 0, 0), 170), ((0.6, 0.8, 0), -95), ((0, 1, 0), 180)]:
        R = turn(np.array(axis, float), degrees)
        w, x, y, z = ms.quaternion_of(R)
        back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(back, R, atol=1e-9)
    a, b = ms.quaternion_of(turn((0, 0, 1), 40)), ms.quaternion_of(turn((1, 0, 0), 25))
    product = ms.multiply(a, b[None])[0]
    assert np.allclose(np.abs(product), np.abs(ms.quaternion_of(turn((0, 0, 1), 40) @ turn((1, 0, 0), 25))), atol=1e-9)


def test_frame_is_metres_y_up_from_the_rooms_centre_on_the_floor():
    frame = ms.Frame(room(), floor_height=5.0)
    # The room's centre on the floor is the origin; 2 scene units are a metre; scene z is up.
    assert np.allclose(frame.scene_to_viewer([10, 20, 5]), [0, 0, 0])
    assert np.allclose(frame.scene_to_viewer([12, 20, 5]), [1, 0, 0])
    assert np.allclose(frame.scene_to_viewer([10, 20, 9]), [0, 2, 0])
    assert np.allclose(frame.scene_to_viewer([10, 22, 5]), [0, 0, -1])       # scene y runs away from the viewer
    assert np.isclose(np.linalg.det(frame.rotation), 1.0)


def test_quads_face_into_the_room_and_map_texels_without_a_flip():
    from surface_fill import Surface

    frame = ms.Frame(room(), floor_height=0.0)
    floor = Surface("floor", np.array([6.0, 14.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                    np.array([0, 0, 1.0]), cols=80, rows=120, cell=0.1)
    wall = Surface("wall-W0", np.array([6.0, 14.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                   np.array([0, 1.0, 0]), cols=80, rows=50, cell=0.1)
    for surface in (floor, wall):
        positions, normals, uvs, indices = ms.quad(surface, frame)
        a, b, c = positions[indices[:3]]
        assert np.cross(b - a, c - a) @ normals[0] > 0, "the first triangle faces the room"
        a, b, c = positions[indices[3:]]
        assert np.cross(b - a, c - a) @ normals[0] > 0, "and so does the second"
        assert np.allclose(positions[0], frame.scene_to_viewer(surface.origin)) and tuple(uvs[0]) == (0, 0)
        far = surface.origin + surface.cols * surface.cell * surface.u + surface.rows * surface.cell * surface.v
        assert np.allclose(positions[3], frame.scene_to_viewer(far)) and tuple(uvs[3]) == (1, 1)
    assert np.allclose(ms.quad(floor, frame)[1][0], [0, 1, 0]), "the floor's normal is up"


def test_glb_is_a_valid_container_with_the_images_inside():
    from surface_fill import Surface

    frame = ms.Frame(room(), floor_height=0.0)
    surface = Surface("floor", np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 8, 8, 0.5)
    jpeg = b"\xff\xd8fakejpeg\xff\xd9"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "shell.glb"
        ms.write_glb(path, [{"name": "floor", "quad": ms.quad(surface, frame), "jpeg": jpeg, "roughness": 0.4},
                            {"name": "wall-W1", "quad": ms.quad(surface, frame), "jpeg": jpeg, "roughness": 0.9}])
        data = path.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    json_length, json_kind = struct.unpack("<II", data[12:20])
    assert json_kind == 0x4E4F534A and json_length % 4 == 0
    document = json.loads(data[20:20 + json_length])
    bin_length, bin_kind = struct.unpack("<II", data[20 + json_length:28 + json_length])
    assert bin_kind == 0x004E4942 and bin_length == document["buffers"][0]["byteLength"]
    assert [m["name"] for m in document["meshes"]] == ["floor", "wall-W1"]
    assert document["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.4
    binary = data[28 + json_length:]
    image = document["bufferViews"][document["images"][0]["bufferView"]]
    assert binary[image["byteOffset"]:image["byteOffset"] + image["byteLength"]] == jpeg
    for accessor in document["accessors"]:
        assert document["bufferViews"][accessor["bufferView"]]["byteOffset"] % 4 == 0


def test_a_piece_file_is_32_bytes_a_gaussian_in_the_viewers_frame():
    from splat_tools import SPLAT_DTYPE

    arr = np.zeros(3, dtype=SPLAT_DTYPE)
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = np.log(0.2)     # 0.2 scene units = 0.1 m
    arr["opacity"] = [4.0, 0.0, -4.0]
    arr["rot_0"] = 1.0
    arr["f_dc_0"] = (1.0 - 0.5) / ms.SH_C0                              # pure red
    frame = ms.Frame(room(world=turn((0, 0, 1), 90)), floor_height=0.0)
    scene = np.array([[12.0, 20.0, 1.0], [10.0, 20.0, 0.0], [10.0, 24.0, 2.0]])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "piece.splat"
        ms.write_piece(path, arr, frame, scene, anchor=np.array([1.0, 0.0, 0.0]))
        raw = path.read_bytes()
    assert len(raw) == 3 * 32
    rows = np.frombuffer(raw, dtype=[("position", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rotation", "u1", 4)])
    assert rows["rgba"][0][3] > rows["rgba"][1][3] > rows["rgba"][2][3], "most solid first"
    assert np.allclose(rows["position"][0], [0.0, 0.5, 0.0], atol=1e-6), "relative to the anchor, metres, y up"
    assert np.allclose(rows["scale"], 0.1, atol=1e-6)
    assert tuple(rows["rgba"][0][:3]) == (255, 127, 127)
    # The Gaussians' rotations carry the frame's turn: no longer the identity (w = 1 -> byte 255).
    assert rows["rotation"][0][0] < 250


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
