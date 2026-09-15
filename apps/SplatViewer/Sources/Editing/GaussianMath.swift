import Foundation
import simd
import SplatIO

/// Moving, turning, resizing and repainting Gaussians.
enum GaussianMath {
    static func rotationZ(_ angle: Float) -> simd_float3x3 {
        let c = cos(angle), s = sin(angle)
        return simd_float3x3(columns: (SIMD3(c, s, 0), SIMD3(-s, c, 0), SIMD3(0, 0, 1)))
    }

    /// Eigen-decomposition of a symmetric 3×3 matrix (cyclic Jacobi): eigenvalues and a
    /// rotation whose columns are the eigenvectors.
    static func symmetricEigen(_ m: simd_float3x3) -> (values: SIMD3<Float>, vectors: simd_float3x3) {
        var a = [[Double]](repeating: [0, 0, 0], count: 3)
        for row in 0..<3 { for col in 0..<3 { a[row][col] = Double(m[col][row]) } }
        var v: [[Double]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for _ in 0..<16 {
            let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2]
            if off < 1e-24 { break }
            for p in 0..<2 {
                for q in (p + 1)..<3 where abs(a[p][q]) > 1e-30 {
                    let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot(), s = t * c
                    for k in 0..<3 {
                        let akp = a[k][p], akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for k in 0..<3 {
                        let apk = a[p][k], aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for k in 0..<3 {
                        let vkp = v[k][p], vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }
        var vectors = simd_float3x3(columns: (
            SIMD3(Float(v[0][0]), Float(v[1][0]), Float(v[2][0])),
            SIMD3(Float(v[0][1]), Float(v[1][1]), Float(v[2][1])),
            SIMD3(Float(v[0][2]), Float(v[1][2]), Float(v[2][2]))))
        if simd_determinant(vectors) < 0 { vectors.columns.2 = -vectors.columns.2 }
        return (SIMD3(Float(a[0][0]), Float(a[1][1]), Float(a[2][2])), vectors)
    }

    /// An object's Gaussians after `placement`: scaled about the middle of its footprint
    /// on the floor, turned about up, and shifted. Positions and shapes are in the splat's
    /// frame; the transform is defined in the room's scene frame.
    static func place(_ points: [SplatPoint], object: RoomModel.Object, placement: Placement,
                      room: RoomModel) -> [SplatPoint] {
        let pivot = SIMD3(object.centre.x, object.centre.y, object.min.z)
        let turn = rotationZ(placement.yaw)
        let stretch = simd_float3x3(diagonal: placement.scale)
        let linearScene = turn * stretch
        let world = room.world, back = room.world.transpose
        let linear = back * linearScene * world                   // the same map in the splat frame
        let uniform = abs(placement.scale.x - placement.scale.y) < 1e-4 && abs(placement.scale.x - placement.scale.z) < 1e-4
        let turnSplat = simd_quaternion(back * turn * world)
        let target = pivot + SIMD3(placement.offset.x, placement.offset.y, 0)
        return points.map { point in
            var moved = point
            let scene = world * point.position
            moved.position = back * (target + linearScene * (scene - pivot))
            if uniform {
                moved.rotation = simd_normalize(turnSplat * point.rotation)
                moved.scale = .linearFloat(point.scale.asLinearFloat * placement.scale.x)
            } else {
                let r = simd_matrix3x3(point.rotation)
                let s = point.scale.asLinearFloat
                let covariance = r * simd_float3x3(diagonal: s * s) * r.transpose
                let stretched = linear * covariance * linear.transpose
                let (values, vectors) = symmetricEigen(stretched)
                moved.rotation = simd_normalize(simd_quaternion(vectors))
                moved.scale = .linearFloat(SIMD3(sqrt(max(values.x, 1e-14)), sqrt(max(values.y, 1e-14)),
                                                  sqrt(max(values.z, 1e-14))))
            }
            return moved
        }
    }

    struct PaintReference: Sendable {
        var colour: SIMD3<Float>
        var luminance: Float
    }

    static func luminance(_ colour: SIMD3<Float>) -> Float {
        simd_dot(colour, SIMD3(0.2126, 0.7152, 0.0722))
    }

    /// The wall's own paint: the most common colour among the Gaussians lying flat on the
    /// wall, which leaves out curtains, shelves and pictures standing off it.
    static func paintReference(_ points: [SplatPoint], wall: RoomModel.Wall, room: RoomModel) -> PaintReference {
        var weights = [Float](repeating: 0, count: 512)
        var sums = [SIMD3<Float>](repeating: .zero, count: 512)
        for flatOnly in [true, false] {
            for point in points {
                if flatOnly {
                    let distance = abs(simd_dot(room.toScene(point.position) - wall.centre, wall.inward))
                    guard distance < 0.025 * room.metre else { continue }
                }
                let colour = simd_clamp(point.color.asSRGBFloat, SIMD3(repeating: 0), SIMD3(repeating: 1))
                let q = SIMD3<Int>(min(Int(colour.x * 8), 7), min(Int(colour.y * 8), 7), min(Int(colour.z * 8), 7))
                let weight = point.opacity.asLinearFloat
                weights[q.x * 64 + q.y * 8 + q.z] += weight
                sums[q.x * 64 + q.y * 8 + q.z] += colour * weight
            }
            if weights.reduce(0, +) > 1 { break }
        }
        guard let best = weights.indices.max(by: { weights[$0] < weights[$1] }), weights[best] > 0 else {
            return PaintReference(colour: SIMD3(repeating: 0.8), luminance: 0.8)
        }
        let bx = best / 64, by = (best / 8) % 8, bz = best % 8
        var sum = SIMD3<Float>.zero, weight: Float = 0
        for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
            let x = bx + dx, y = by + dy, z = bz + dz
            guard (0..<8).contains(x), (0..<8).contains(y), (0..<8).contains(z) else { continue }
            sum += sums[x * 64 + y * 8 + z]
            weight += weights[x * 64 + y * 8 + z]
        } } }
        let colour = sum / max(weight, 1e-6)
        return PaintReference(colour: colour, luminance: max(luminance(colour), 0.05))
    }

    /// Repaints Gaussians that carry the wall's paint (their colour close to the reference
    /// in hue, and, for the wall's own Gaussians, lying on it) and keeps their light and
    /// shade. Anything else on the wall keeps its colour.
    static func paint(_ points: [SplatPoint], colour: SIMD3<Float>, reference: PaintReference,
                      wall: RoomModel.Wall?, room: RoomModel) -> [SplatPoint] {
        let referenceChroma = reference.colour / reference.luminance
        return points.map { point in
            let original = simd_clamp(point.color.asSRGBFloat, SIMD3(repeating: 0), SIMD3(repeating: 1))
            let lum = luminance(original)
            let chroma = original / max(lum, 0.05)
            let ratio = lum / reference.luminance
            var weight = 1 - SurfacePatch.smoothstep(0.08, 0.2, simd_length(chroma - referenceChroma))
            weight *= SurfacePatch.smoothstep(0.12, 0.3, ratio) * (1 - SurfacePatch.smoothstep(1.6, 2.2, ratio))
            if let wall {
                let distance = abs(simd_dot(room.toScene(point.position) - wall.centre, wall.inward))
                weight *= 1 - SurfacePatch.smoothstep(0.045 * room.metre, 0.08 * room.metre, distance)
            }
            guard weight > 0.01 else { return point }
            let shade = min(max(ratio, 0.35), 1.5)
            let target = simd_clamp(colour * shade, SIMD3(repeating: 0), SIMD3(repeating: 1))
            let mixed = original + (target - original) * weight
            var painted = point
            switch point.color {
            case .sRGBUInt8:
                painted.color = .sRGBUInt8(SIMD3<UInt8>((mixed * 255).rounded(.toNearestOrAwayFromZero)))
            case .sphericalHarmonicFloat(let bands):
                var faded = bands.map { $0 * (1 - weight) }
                faded[0] = (mixed - 0.5) / SplatPoint.Color.SH_C0
                painted.color = .sphericalHarmonicFloat(faded)
            }
            return painted
        }
    }
}

extension SIMD3 where Scalar == Float {
    /// sRGB 0…1 from "#RRGGBB".
    init?(hex: String) {
        let digits = hex.trimmingCharacters(in: CharacterSet(charactersIn: "# "))
        guard digits.count == 6, let value = UInt32(digits, radix: 16) else { return nil }
        self.init(Float((value >> 16) & 0xFF) / 255, Float((value >> 8) & 0xFF) / 255, Float(value & 0xFF) / 255)
    }

    var hex: String {
        let c = simd_clamp(self, SIMD3(repeating: 0), SIMD3(repeating: 1)) * 255
        return String(format: "#%02X%02X%02X", Int(c.x.rounded()), Int(c.y.rounded()), Int(c.z.rounded()))
    }
}
