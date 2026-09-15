import Foundation
import simd

/// What stage 3 measured about the room (spaces/<name>/shapes.json and densify.json),
/// in the scene frame: the splat's frame turned so walls run along x and y and z is up.
/// scene = world * splat, splat = worldᵀ * scene.
struct RoomModel {
    struct Object: Identifiable {
        let id: String          // "B4", the box's index in shapes.json
        let label: String       // bed, wardrobe, …
        let min: SIMD3<Float>
        let max: SIMD3<Float>

        var centre: SIMD2<Float> { SIMD2((min.x + max.x) / 2, (min.y + max.y) / 2) }
        var size: SIMD3<Float> { max - min }
    }

    struct Wall: Identifiable {
        let id: String          // "W2", the plane's index in shapes.json
        let centre: SIMD3<Float>
        let along: SIMD3<Float> // unit, horizontal
        let inward: SIMD3<Float>// unit, horizontal, facing into the room
        let halfLength: Float
        let halfHeight: Float
    }

    let world: simd_float3x3
    let metre: Float
    let floorZ: Float
    let height: Float
    let centre: SIMD2<Float>
    let half: SIMD2<Float>
    let objects: [Object]
    let walls: [Wall]

    /// The scene's up direction in the splat's frame.
    var up: SIMD3<Float> { simd_normalize(world.transpose * SIMD3(0, 0, 1)) }

    func toScene(_ p: SIMD3<Float>) -> SIMD3<Float> { world * p }
    func toSplat(_ p: SIMD3<Float>) -> SIMD3<Float> { world.transpose * p }

    func object(_ id: String) -> Object? { objects.first { $0.id == id } }
    func wall(_ id: String) -> Wall? { walls.first { $0.id == id } }

    static func load(besides splat: URL) -> RoomModel? {
        let folder = splat.deletingLastPathComponent()
        guard let shapesData = try? Data(contentsOf: folder.appendingPathComponent("shapes.json")),
              let shapes = try? JSONSerialization.jsonObject(with: shapesData) as? [String: Any],
              let metaData = try? Data(contentsOf: folder.appendingPathComponent("densify.json")),
              let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              let metre = (meta["colmap_units_per_metre"] as? NSNumber)?.floatValue, metre > 0,
              let worldRows = shapes["world"] as? [[NSNumber]], worldRows.count == 3,
              let room = shapes["room"] as? [String: Any],
              let level = shapes["room_level"] as? [String: Any],
              let floorZ = (level["floor_z"] as? NSNumber)?.floatValue,
              let height = (level["height"] as? NSNumber)?.floatValue else { return nil }

        func vector(_ value: Any?) -> SIMD3<Float>? {
            guard let list = value as? [NSNumber], list.count >= 3 else { return nil }
            return SIMD3(list[0].floatValue, list[1].floatValue, list[2].floatValue)
        }
        let rows = worldRows.map { SIMD3($0[0].floatValue, $0[1].floatValue, $0[2].floatValue) }
        let world = simd_float3x3(columns: (rows[0], rows[1], rows[2])).transpose
        let centreList = (room["center"] as? [NSNumber]) ?? [0, 0]
        let centre = SIMD2(centreList[0].floatValue, centreList[1].floatValue)
        let half = SIMD2((room["half_u"] as? NSNumber)?.floatValue ?? 1, (room["half_v"] as? NSNumber)?.floatValue ?? 1)

        var objects: [Object] = []
        for (index, box) in ((shapes["boxes"] as? [[String: Any]]) ?? []).enumerated() {
            guard (box["build"] as? Bool) ?? true, let lo = vector(box["min"]), let hi = vector(box["max"]) else { continue }
            objects.append(Object(id: "B\(index)", label: (box["label"] as? String) ?? "object",
                                  min: SIMD3(lo.x, lo.y, max(lo.z, floorZ)), max: hi))
        }
        var walls: [Wall] = []
        for (index, plane) in ((shapes["planes"] as? [[String: Any]]) ?? []).enumerated() {
            guard (plane["label"] as? String) == "wall", (plane["build"] as? Bool) ?? true,
                  let c = vector(plane["center"]), var a = vector(plane["axis_a"]), var n = vector(plane["normal"]),
                  let halfA = (plane["half_a"] as? NSNumber)?.floatValue,
                  let halfB = (plane["half_b"] as? NSNumber)?.floatValue else { continue }
            a.z = 0; n.z = 0
            guard simd_length(a) > 1e-4, simd_length(n) > 1e-4 else { continue }
            a = simd_normalize(a); n = simd_normalize(n)
            if simd_dot(SIMD3(centre.x, centre.y, c.z) - c, n) < 0 { n = -n }
            walls.append(Wall(id: "W\(index)", centre: c, along: a, inward: n, halfLength: halfA, halfHeight: halfB))
        }
        return RoomModel(world: world, metre: metre, floorZ: floorZ, height: height, centre: centre, half: half,
                         objects: objects, walls: walls)
    }
}

/// Where an object stands after editing, relative to where it was filmed.
struct Placement: Codable, Equatable {
    var offset = SIMD2<Float>.zero   // scene units along the floor
    var yaw: Float = 0               // radians about up
    var scale = SIMD3<Float>(1, 1, 1)// width (x), depth (y), height (z)

    var isIdentity: Bool { offset == .zero && yaw == 0 && scale == SIMD3(1, 1, 1) }
}

/// A box standing on the floor, turned about up: what selection, handles and hit tests use.
struct FloorBox {
    var centre: SIMD2<Float>
    var half: SIMD2<Float>
    var bottom: Float
    var top: Float
    var yaw: Float

    var axisX: SIMD2<Float> { SIMD2(cos(yaw), sin(yaw)) }
    var axisY: SIMD2<Float> { SIMD2(-sin(yaw), cos(yaw)) }
    /// Half-extents of the turned footprint along the scene's x and y.
    var reach: SIMD2<Float> { simd_abs(axisX) * half.x + simd_abs(axisY) * half.y }

    func corner(_ sx: Float, _ sy: Float, _ z: Float) -> SIMD3<Float> {
        let xy = centre + axisX * half.x * sx + axisY * half.y * sy
        return SIMD3(xy.x, xy.y, z)
    }

    /// Bottom four (counter-clockwise from -x -y), then the top four above them.
    var corners: [SIMD3<Float>] {
        let signs: [(Float, Float)] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        return signs.map { corner($0.0, $0.1, bottom) } + signs.map { corner($0.0, $0.1, top) }
    }

    /// Distance along a scene-frame ray to where it enters the box, if it does.
    func hit(origin: SIMD3<Float>, direction: SIMD3<Float>) -> Float? {
        // Into the box's own frame, where it is axis-aligned.
        func local(_ v: SIMD3<Float>, point: Bool) -> SIMD3<Float> {
            let xy = SIMD2(v.x, v.y) - (point ? centre : .zero)
            return SIMD3(simd_dot(xy, axisX), simd_dot(xy, axisY), v.z - (point ? (bottom + top) / 2 : 0))
        }
        let o = local(origin, point: true), d = local(direction, point: false)
        let extent = SIMD3(half.x, half.y, (top - bottom) / 2)
        var tNear = -Float.infinity, tFar = Float.infinity
        for axis in 0..<3 {
            if abs(d[axis]) < 1e-9 {
                if abs(o[axis]) > extent[axis] { return nil }
            } else {
                var t1 = (-extent[axis] - o[axis]) / d[axis], t2 = (extent[axis] - o[axis]) / d[axis]
                if t1 > t2 { swap(&t1, &t2) }
                tNear = max(tNear, t1); tFar = min(tFar, t2)
                if tNear > tFar { return nil }
            }
        }
        guard tFar > 0 else { return nil }
        return max(tNear, 0)
    }
}

extension RoomModel.Object {
    func box(_ placement: Placement) -> FloorBox {
        FloorBox(centre: centre + placement.offset,
                 half: SIMD2(size.x, size.y) / 2 * SIMD2(placement.scale.x, placement.scale.y),
                 bottom: min.z, top: min.z + size.z * placement.scale.z, yaw: placement.yaw)
    }
}
