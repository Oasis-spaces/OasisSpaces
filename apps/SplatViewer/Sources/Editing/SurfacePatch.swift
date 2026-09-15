import Foundation
import simd
import SplatIO

/// Covers what a moved or removed object had hidden. The capture never saw the floor under
/// a bed or the wall behind a wardrobe, so taking the object away leaves a hole. A patch
/// lays thin Gaussians over the bare part of that floor or wall, just behind the surface
/// so real Gaussians still win where there are any, coloured by carrying the surface
/// around the hole inward and easing toward its typical colour in the middle.
enum SurfacePatch {
    /// The floor under an object and any wall it stood against, in the splat's frame.
    static func patches(for object: RoomModel.Object, room: RoomModel, floor: [SplatPoint],
                        walls: [String: [SplatPoint]]) -> (floor: [SplatPoint], walls: [String: [SplatPoint]]) {
        let m = room.metre
        let margin = 0.06 * m
        let floorPatch = build(origin: SIMD3(0, 0, room.floorZ), u: SIMD3(1, 0, 0), v: SIMD3(0, 1, 0),
                               lo: SIMD2(object.min.x, object.min.y) - margin,
                               hi: SIMD2(object.max.x, object.max.y) + margin,
                               offset: SIMD3(0, 0, -0.004 * m), band: 0.04 * m, surface: floor, room: room)
        var wallPatches: [String: [SplatPoint]] = [:]
        let footprint = [SIMD2(object.min.x, object.min.y), SIMD2(object.max.x, object.min.y),
                         SIMD2(object.max.x, object.max.y), SIMD2(object.min.x, object.max.y)]
        for wall in room.walls {
            guard let surface = walls[wall.id] else { continue }
            let offsets = footprint.map { SIMD3($0.x, $0.y, wall.centre.z) - wall.centre }
            let nearest = offsets.map { simd_dot($0, wall.inward) }.min() ?? .infinity
            guard nearest < 0.35 * m, nearest > -0.3 * m else { continue }
            let along = offsets.map { simd_dot($0, wall.along) }
            let from = max(along.min()! - margin, -wall.halfLength)
            let to = min(along.max()! + margin, wall.halfLength)
            guard to - from > 0.05 * m else { continue }
            let top = min(object.max.z + 0.08 * m, room.floorZ + room.height)
            let patch = build(origin: SIMD3(wall.centre.x, wall.centre.y, 0), u: wall.along, v: SIMD3(0, 0, 1),
                              lo: SIMD2(from, room.floorZ), hi: SIMD2(to, top),
                              offset: -wall.inward * 0.004 * m, band: 0.06 * m, surface: surface, room: room)
            if !patch.isEmpty { wallPatches[wall.id] = patch }
        }
        return (floorPatch, wallPatches)
    }

    /// Gaussians over the bare cells of the rectangle `lo`…`hi` (in `u`, `v` from `origin`,
    /// scene frame) on the plane through `origin`. `surface` holds Gaussians on that plane.
    static func build(origin: SIMD3<Float>, u: SIMD3<Float>, v: SIMD3<Float>, lo: SIMD2<Float>, hi: SIMD2<Float>,
                      offset: SIMD3<Float>, band: Float, surface: [SplatPoint], room: RoomModel) -> [SplatPoint] {
        let m = room.metre
        let normal = simd_normalize(simd_cross(u, v))
        let ring = 0.4 * m, coverCell = 0.05 * m, sampleCell = 0.08 * m, spacing = 0.025 * m
        let extent = hi - lo
        guard extent.x > 0, extent.y > 0 else { return [] }
        let coverW = max(1, Int(ceil(extent.x / coverCell))), coverH = max(1, Int(ceil(extent.y / coverCell)))
        var cover = [UInt16](repeating: 0, count: coverW * coverH)

        // Around the hole: colour sums on a coarse grid (cells counted from `lo - ring`).
        let sampleOrigin = lo - ring
        let sampleW = Int(ceil((extent.x + 2 * ring) / sampleCell)), sampleH = Int(ceil((extent.y + 2 * ring) / sampleCell))
        var sums = [SIMD3<Float>](repeating: .zero, count: sampleW * sampleH)
        var counts = [Int](repeating: 0, count: sampleW * sampleH)
        var ringColours: [SIMD3<Float>] = []

        for point in surface where point.opacity.asLinearFloat > 0.3 {
            let p = room.toScene(point.position) - origin
            guard abs(simd_dot(p, normal)) < band else { continue }
            let uv = SIMD2(simd_dot(p, u), simd_dot(p, v))
            let s = (uv - sampleOrigin) / sampleCell
            guard s.x >= 0, s.y >= 0, Int(s.x) < sampleW, Int(s.y) < sampleH else { continue }
            if uv.x >= lo.x, uv.x < hi.x, uv.y >= lo.y, uv.y < hi.y {
                let c = (uv - lo) / coverCell
                cover[min(Int(c.y), coverH - 1) * coverW + min(Int(c.x), coverW - 1)] &+= 1
            } else {
                let colour = simd_clamp(point.color.asSRGBFloat, SIMD3(repeating: 0), SIMD3(repeating: 1))
                let index = Int(s.y) * sampleW + Int(s.x)
                sums[index] += colour
                counts[index] += 1
                ringColours.append(colour)
            }
        }
        guard ringColours.count >= 12 else { return [] }
        func channelMedian(_ channel: Int) -> Float {
            let values = ringColours.map { $0[channel] }.sorted()
            return values[values.count / 2]
        }
        let median = SIMD3(channelMedian(0), channelMedian(1), channelMedian(2))

        // For each sample cell over the hole, the mean colour of the nearest cell around it.
        var nearestColour = [SIMD3<Float>?](repeating: nil, count: sampleW * sampleH)
        let maxRadius = max(sampleW, sampleH)
        for y in 0..<sampleH {
            for x in 0..<sampleW {
                let cellUV = sampleOrigin + (SIMD2(Float(x), Float(y)) + 0.5) * sampleCell
                guard cellUV.x > lo.x - sampleCell, cellUV.x < hi.x + sampleCell,
                      cellUV.y > lo.y - sampleCell, cellUV.y < hi.y + sampleCell else { continue }
                var found: SIMD3<Float>?
                var bestDistance = Int.max
                for radius in 0..<maxRadius {
                    if let _ = found, radius * radius > bestDistance { break }
                    for dy in -radius...radius {
                        for dx in -radius...radius where max(abs(dx), abs(dy)) == radius {
                            let nx = x + dx, ny = y + dy
                            guard nx >= 0, ny >= 0, nx < sampleW, ny < sampleH else { continue }
                            let index = ny * sampleW + nx
                            guard counts[index] >= 3 else { continue }
                            let distance = dx * dx + dy * dy
                            if distance < bestDistance {
                                bestDistance = distance
                                found = sums[index] / Float(counts[index])
                            }
                        }
                    }
                }
                nearestColour[y * sampleW + x] = found
            }
        }

        let frame = room.world.transpose * simd_float3x3(columns: (u, v, normal))
        let rotation = simd_normalize(simd_quaternion(frame))
        let scale = SIMD3(spacing * 0.75, spacing * 0.75, spacing * 0.08)
        var rng = SystemRandomNumberGenerator()
        var out: [SplatPoint] = []
        let countU = Int(ceil(extent.x / spacing)), countV = Int(ceil(extent.y / spacing))
        out.reserveCapacity(countU * countV / 2)
        for i in 0..<countU {
            for j in 0..<countV {
                let jitter = SIMD2(Float.random(in: -0.3...0.3, using: &rng), Float.random(in: -0.3...0.3, using: &rng))
                let uv = simd_clamp(lo + (SIMD2(Float(i), Float(j)) + 0.5 + jitter) * spacing, lo, hi - 1e-4)
                let c = (uv - lo) / coverCell
                // Real surface here: leave it be.
                if cover[min(Int(c.y), coverH - 1) * coverW + min(Int(c.x), coverW - 1)] >= 3 { continue }
                let s = (uv - sampleOrigin) / sampleCell
                let near = nearestColour[min(Int(s.y), sampleH - 1) * sampleW + min(Int(s.x), sampleW - 1)] ?? median
                let edge = min(uv.x - lo.x, hi.x - uv.x, uv.y - lo.y, hi.y - uv.y)
                let t = smoothstep(0, 0.35 * m, edge) * 0.8
                let grain = 1 + Float.random(in: -0.025...0.025, using: &rng)
                let colour = simd_clamp((near + (median - near) * t) * grain, SIMD3(repeating: 0), SIMD3(repeating: 1))
                let position = origin + u * uv.x + v * uv.y + offset
                out.append(SplatPoint(position: room.toSplat(position),
                                      color: .sRGBUInt8(SIMD3<UInt8>(colour * 255)),
                                      opacity: .linearFloat(0.99),
                                      scale: .linearFloat(scale),
                                      rotation: rotation))
            }
        }
        return out
    }

    static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = min(max((x - edge0) / max(edge1 - edge0, 1e-6), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
