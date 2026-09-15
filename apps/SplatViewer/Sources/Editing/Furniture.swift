import Foundation
import simd
import SplatIO

/// Furniture that can be added to a room, the same pieces and sizes as the website's
/// editor (room.js), built from boxes, cylinders and blobs and turned into Gaussians
/// lying on their surfaces.
enum FurnitureKind: String, CaseIterable, Codable, Identifiable {
    case sofa, armchair, chair, bed, sidetable, desk, wardrobe, shelf, lamp, plant, rug

    var id: String { rawValue }

    var label: String {
        switch self {
        case .sofa: "Sofa"
        case .armchair: "Armchair"
        case .chair: "Chair"
        case .bed: "Bed"
        case .sidetable: "Bedside table"
        case .desk: "Desk"
        case .wardrobe: "Wardrobe"
        case .shelf: "Open shelf"
        case .lamp: "Floor lamp"
        case .plant: "Plant"
        case .rug: "Rug"
        }
    }

    var systemImage: String {
        switch self {
        case .sofa: "sofa.fill"
        case .armchair: "chair.lounge.fill"
        case .chair: "chair.fill"
        case .bed: "bed.double.fill"
        case .sidetable: "square.stack.3d.up.fill"
        case .desk: "table.furniture.fill"
        case .wardrobe: "cabinet.fill"
        case .shelf: "books.vertical.fill"
        case .lamp: "lamp.floor.fill"
        case .plant: "leaf.fill"
        case .rug: "rectangle.fill"
        }
    }

    /// Width (x), depth (back at +y), height, in metres.
    var size: SIMD3<Float> {
        switch self {
        case .sofa: SIMD3(2.0, 0.9, 0.96)
        case .armchair: SIMD3(0.85, 0.85, 0.92)
        case .chair: SIMD3(0.48, 0.5, 0.91)
        case .bed: SIMD3(1.6, 2.0, 1.0)
        case .sidetable: SIMD3(0.5, 0.45, 0.56)
        case .desk: SIMD3(1.3, 0.65, 0.755)
        case .wardrobe: SIMD3(1.2, 0.6, 2.1)
        case .shelf: SIMD3(1.0, 0.32, 1.7)
        case .lamp: SIMD3(0.46, 0.46, 1.58)
        case .plant: SIMD3(0.5, 0.5, 0.95)
        case .rug: SIMD3(2.6, 1.8, 0.015)
        }
    }

    /// The material the colour picker changes.
    var primaryRole: Role {
        switch self {
        case .sofa, .armchair, .bed: .fabric
        case .chair, .wardrobe, .sidetable: .wood2
        case .desk, .shelf: .wood
        case .lamp: .shade
        case .plant: .leaf
        case .rug: .rug
        }
    }

    enum Role {
        case wood, wood2, mattress, fabric, pillow, shade, metal, pot, leaf, rug, accent

        /// The website's "original" theme.
        var colour: SIMD3<Float> {
            let hex: String = switch self {
            case .wood: "#6E4F35"
            case .wood2: "#8A6A4B"
            case .mattress: "#F3EFE7"
            case .fabric: "#8E9DA6"
            case .pillow: "#F7F4EE"
            case .shade: "#F1E6CF"
            case .metal: "#2A2926"
            case .pot: "#A9754E"
            case .leaf: "#4D7A4C"
            case .rug: "#CFC3B0"
            case .accent: "#3F5C52"
            }
            return SIMD3(hex: hex)!
        }
    }

    /// A part in the website's layout: x across, y up from the floor, z depth with the back at -z.
    enum Part {
        case box(Role, w: Float, h: Float, d: Float, x: Float, y: Float, z: Float)
        case cylinder(Role, top: Float, bottom: Float, h: Float, x: Float, y: Float, z: Float)
        case blob(Role, r: Float, x: Float, y: Float, z: Float)
    }

    var parts: [Part] {
        let legs4: (Float, Float, Float, Role, Float) -> [Part] = { dx, dz, h, role, t in
            [(-dx, -dz), (dx, -dz), (-dx, dz), (dx, dz)].map { .box(role, w: t, h: h, d: t, x: $0.0, y: 0, z: $0.1) }
        }
        switch self {
        case .bed:
            return [.box(.wood, w: 1.7, h: 1.0, d: 0.08, x: 0, y: 0, z: -0.96),
                    .box(.wood2, w: 1.6, h: 0.26, d: 1.9, x: 0, y: 0, z: 0.02),
                    .box(.mattress, w: 1.5, h: 0.22, d: 1.84, x: 0, y: 0.26, z: 0.02),
                    .box(.fabric, w: 1.54, h: 0.09, d: 1.25, x: 0, y: 0.47, z: 0.32),
                    .box(.pillow, w: 0.62, h: 0.13, d: 0.42, x: -0.4, y: 0.48, z: -0.62),
                    .box(.pillow, w: 0.62, h: 0.13, d: 0.42, x: 0.4, y: 0.48, z: -0.62)]
        case .wardrobe:
            return [.box(.wood2, w: 1.2, h: 2.1, d: 0.6, x: 0, y: 0, z: 0),
                    .box(.metal, w: 0.012, h: 1.9, d: 0.012, x: 0, y: 0.1, z: 0.3),
                    .box(.metal, w: 0.02, h: 0.18, d: 0.02, x: -0.06, y: 0.95, z: 0.31),
                    .box(.metal, w: 0.02, h: 0.18, d: 0.02, x: 0.06, y: 0.95, z: 0.31)]
        case .sidetable:
            return [.box(.wood, w: 0.5, h: 0.04, d: 0.45, x: 0, y: 0.52, z: 0),
                    .box(.wood2, w: 0.46, h: 0.16, d: 0.4, x: 0, y: 0.34, z: 0)]
                + legs4(0.21, 0.18, 0.34, .metal, 0.025)
        case .lamp:
            return [.cylinder(.metal, top: 0.16, bottom: 0.17, h: 0.02, x: 0, y: 0, z: 0),
                    .cylinder(.metal, top: 0.012, bottom: 0.012, h: 1.28, x: 0, y: 0.02, z: 0),
                    .cylinder(.shade, top: 0.17, bottom: 0.23, h: 0.32, x: 0, y: 1.26, z: 0)]
        case .plant:
            return [.cylinder(.pot, top: 0.17, bottom: 0.13, h: 0.32, x: 0, y: 0, z: 0),
                    .blob(.leaf, r: 0.22, x: 0.02, y: 0.6, z: 0),
                    .blob(.leaf, r: 0.18, x: -0.14, y: 0.72, z: 0.08),
                    .blob(.leaf, r: 0.17, x: 0.15, y: 0.76, z: -0.07),
                    .blob(.leaf, r: 0.14, x: 0, y: 0.84, z: 0.12)]
        case .armchair:
            return legs4(0.36, 0.34, 0.12, .wood, 0.04)
                + [.box(.fabric, w: 0.85, h: 0.34, d: 0.8, x: 0, y: 0.1, z: 0.02),
                   .box(.fabric, w: 0.85, h: 0.5, d: 0.2, x: 0, y: 0.42, z: -0.3),
                   .box(.fabric, w: 0.16, h: 0.22, d: 0.8, x: -0.345, y: 0.42, z: 0.02),
                   .box(.fabric, w: 0.16, h: 0.22, d: 0.8, x: 0.345, y: 0.42, z: 0.02)]
        case .sofa:
            return legs4(0.9, 0.36, 0.12, .wood, 0.04)
                + [.box(.fabric, w: 2.0, h: 0.34, d: 0.85, x: 0, y: 0.1, z: 0.02),
                   .box(.pillow, w: 0.9, h: 0.12, d: 0.7, x: -0.48, y: 0.44, z: 0.05),
                   .box(.pillow, w: 0.9, h: 0.12, d: 0.7, x: 0.48, y: 0.44, z: 0.05),
                   .box(.fabric, w: 2.0, h: 0.52, d: 0.22, x: 0, y: 0.44, z: -0.31),
                   .box(.fabric, w: 0.18, h: 0.22, d: 0.85, x: -0.91, y: 0.44, z: 0.02),
                   .box(.fabric, w: 0.18, h: 0.22, d: 0.85, x: 0.91, y: 0.44, z: 0.02)]
        case .rug:
            return [.box(.rug, w: 2.6, h: 0.015, d: 1.8, x: 0, y: 0, z: 0)]
        case .desk:
            return [.box(.wood, w: 1.3, h: 0.035, d: 0.65, x: 0, y: 0.72, z: 0)]
                + legs4(0.6, 0.28, 0.72, .metal, 0.03)
        case .chair:
            return [.box(.wood2, w: 0.46, h: 0.04, d: 0.46, x: 0, y: 0.45, z: 0),
                    .box(.wood2, w: 0.44, h: 0.42, d: 0.03, x: 0, y: 0.49, z: -0.215)]
                + legs4(0.2, 0.2, 0.45, .metal, 0.025)
        case .shelf:
            return [.box(.wood, w: 0.03, h: 1.7, d: 0.32, x: -0.485, y: 0, z: 0),
                    .box(.wood, w: 0.03, h: 1.7, d: 0.32, x: 0.485, y: 0, z: 0),
                    .box(.wood2, w: 0.98, h: 1.7, d: 0.012, x: 0, y: 0, z: -0.154)]
                + (0..<5).map { .box(.wood, w: 1.0, h: 0.03, d: 0.32, x: 0, y: Float($0) * 0.415, z: 0) }
                + [.box(.accent, w: 0.05, h: 0.28, d: 0.2, x: -0.3, y: 0.445, z: 0.02),
                   .box(.pot, w: 0.06, h: 0.24, d: 0.2, x: -0.23, y: 0.445, z: 0.02),
                   .box(.fabric, w: 0.22, h: 0.22, d: 0.22, x: 0.25, y: 0.86, z: 0),
                   .cylinder(.pot, top: 0.07, bottom: 0.06, h: 0.12, x: -0.25, y: 1.275, z: 0.02),
                   .blob(.leaf, r: 0.1, x: -0.25, y: 1.46, z: 0.02)]
        }
    }
}

/// A piece of furniture added in the viewer.
struct AddedItem: Codable, Equatable, Identifiable {
    var id = UUID()
    var kind: FurnitureKind
    var centre: SIMD2<Float>      // scene units
    var yaw: Float = 0
    var size: SIMD3<Float>        // metres: width, depth, height
    var colour: String?           // the primary material, "#RRGGBB"

    func box(metre: Float, floorZ: Float) -> FloorBox {
        FloorBox(centre: centre, half: SIMD2(size.x, size.y) * metre / 2, bottom: floorZ,
                 top: floorZ + size.z * metre, yaw: yaw)
    }
}

enum FurnitureBuilder {
    static let spacingMetres: Float = 0.014
    private static let light = simd_normalize(SIMD3<Float>(0.35, -0.45, 0.82))

    /// Gaussians for an added item, in the splat's frame.
    static func points(for item: AddedItem, room: RoomModel) -> [SplatPoint] {
        let m = room.metre
        let base = item.kind.size
        let stretch = item.size / base                     // per axis: width, depth, height
        let spacing = spacingMetres * m
        let turn = GaussianMath.rotationZ(item.yaw)
        let origin = SIMD3(item.centre.x, item.centre.y, room.floorZ)
        let primary = item.colour.flatMap { SIMD3<Float>(hex: $0) }
        var rng = SystemRandomNumberGenerator()
        var out: [SplatPoint] = []

        /// Website coordinates (x across, y up, z depth, back at -z) in metres → scene frame.
        func toScene(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let local = SIMD3(p.x * stretch.x, -p.z * stretch.y, p.y * stretch.z) * m
            return origin + turn * local
        }
        func normalToScene(_ n: SIMD3<Float>) -> SIMD3<Float> {
            simd_normalize(turn * SIMD3(n.x / stretch.x, -n.z / stretch.y, n.y / stretch.z))
        }
        func emit(_ positionWebsite: SIMD3<Float>, _ normalWebsite: SIMD3<Float>, _ role: FurnitureKind.Role) {
            let position = toScene(positionWebsite)
            let normal = normalToScene(normalWebsite)
            let helper: SIMD3<Float> = abs(normal.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
            let t1 = simd_normalize(simd_cross(helper, normal))
            let t2 = simd_cross(normal, t1)
            let frameScene = simd_float3x3(columns: (t1, t2, normal))
            let base = role == item.kind.primaryRole ? (primary ?? role.colour) : role.colour
            let shade = 0.74 + 0.26 * max(0, simd_dot(normal, light))
            let jitter = Float.random(in: -0.015...0.015, using: &rng)
            let colour = simd_clamp(base * shade + jitter, SIMD3(repeating: 0), SIMD3(repeating: 1))
            out.append(SplatPoint(position: room.toSplat(position),
                                  color: .sRGBUInt8(SIMD3<UInt8>(colour * 255)),
                                  opacity: .linearFloat(0.99),
                                  scale: .linearFloat(SIMD3(spacing * 0.8, spacing * 0.8, spacing * 0.08)),
                                  rotation: simd_normalize(simd_quaternion(room.world.transpose * frameScene))))
        }
        /// Grid positions across a span, in metres of the finished (stretched) piece.
        func steps(_ length: Float, _ axisStretch: Float) -> [Float] {
            let real = length * axisStretch
            let count = max(1, Int((real / spacingMetres).rounded()))
            return (0..<count).map { (Float($0) + 0.5) / Float(count) * length - length / 2 }
        }

        for part in item.kind.parts {
            switch part {
            case let .box(role, w, h, d, x, y, z):
                let centre = SIMD3(x, y + h / 2, z)
                for a in steps(w, stretch.x) { for b in steps(d, stretch.y) {
                    emit(centre + SIMD3(a, h / 2, b), SIMD3(0, 1, 0), role)
                    if y > 0.001 { emit(centre + SIMD3(a, -h / 2, b), SIMD3(0, -1, 0), role) }
                } }
                for a in steps(w, stretch.x) { for c in steps(h, stretch.z) {
                    emit(centre + SIMD3(a, c, d / 2), SIMD3(0, 0, 1), role)
                    emit(centre + SIMD3(a, c, -d / 2), SIMD3(0, 0, -1), role)
                } }
                for b in steps(d, stretch.y) { for c in steps(h, stretch.z) {
                    emit(centre + SIMD3(w / 2, c, b), SIMD3(1, 0, 0), role)
                    emit(centre + SIMD3(-w / 2, c, b), SIMD3(-1, 0, 0), role)
                } }
            case let .cylinder(role, top, bottom, h, x, y, z):
                let radius = max(top, bottom) * max(stretch.x, stretch.y)
                let around = max(8, Int((2 * Float.pi * radius / spacingMetres).rounded()))
                for c in steps(h, stretch.z) {
                    let k = (c + h / 2) / h
                    let r = bottom + (top - bottom) * k
                    for i in 0..<around {
                        let angle = Float(i) / Float(around) * 2 * .pi
                        let dir = SIMD3(cos(angle), 0, sin(angle))
                        emit(SIMD3(x, y + h / 2 + c, z) + dir * r, dir, role)
                    }
                }
                let rings = max(1, Int((top * max(stretch.x, stretch.y) / spacingMetres).rounded()))
                for ring in 0..<rings {
                    let r = top * (Float(ring) + 0.5) / Float(rings)
                    let count = max(6, Int((2 * Float.pi * r * max(stretch.x, stretch.y) / spacingMetres).rounded()))
                    for i in 0..<count {
                        let angle = Float(i) / Float(count) * 2 * .pi
                        emit(SIMD3(x + cos(angle) * r, y + h, z + sin(angle) * r), SIMD3(0, 1, 0), role)
                    }
                }
            case let .blob(role, r, x, y, z):
                let area = 4 * Float.pi * r * r * stretch.x * stretch.z
                let count = max(40, Int(area / (spacingMetres * spacingMetres)))
                let golden = Float.pi * (3 - sqrt(5))
                for i in 0..<count {
                    let yy = 1 - 2 * (Float(i) + 0.5) / Float(count)
                    let ring = sqrt(max(0, 1 - yy * yy))
                    let angle = golden * Float(i)
                    let dir = SIMD3(cos(angle) * ring, yy, sin(angle) * ring)
                    emit(SIMD3(x, y, z) + dir * r, dir, role)
                }
            }
        }
        return out
    }
}
