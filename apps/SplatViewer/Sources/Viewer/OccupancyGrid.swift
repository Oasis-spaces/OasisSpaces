import Foundation
import simd

/// Where the splat has solid stuff: a sparse grid counting opaque Gaussians per cell.
/// SceneController uses it to know when the camera has walked into a wall or a piece of
/// furniture, and to look past it.
struct OccupancyGrid: Sendable {
    let cell: Float
    private let counts: [Int64: UInt16]
    /// Opaque Gaussians in a cell for it to count as solid.
    private let solidCount: UInt16 = 4

    init(positions: [SIMD3<Float>], cell: Float) {
        self.cell = cell
        var counts: [Int64: UInt16] = [:]
        counts.reserveCapacity(positions.count / 8)
        for p in positions {
            let key = Self.key(p, cell)
            counts[key, default: 0] &+= 1
        }
        self.counts = counts
    }

    private static func key(_ p: SIMD3<Float>, _ cell: Float) -> Int64 {
        let offset: Int64 = 1 << 20
        let i = Int64(floor(p.x / cell)) + offset, j = Int64(floor(p.y / cell)) + offset
        let k = Int64(floor(p.z / cell)) + offset
        return (i & 0x1F_FFFF) << 42 | (j & 0x1F_FFFF) << 21 | (k & 0x1F_FFFF)
    }

    func isSolid(_ p: SIMD3<Float>) -> Bool {
        (counts[Self.key(p, cell)] ?? 0) >= solidCount
    }

    /// Nothing solid at the point or half a cell around it.
    func isOpen(_ p: SIMD3<Float>) -> Bool {
        let h = cell * 0.5
        let probes: [SIMD3<Float>] = [.zero, SIMD3(h, 0, 0), SIMD3(-h, 0, 0), SIMD3(0, h, 0),
                                      SIMD3(0, -h, 0), SIMD3(0, 0, h), SIMD3(0, 0, -h)]
        return probes.allSatisfy { !isSolid(p + $0) }
    }

    /// Nothing solid on the straight line between two points.
    func isClear(from a: SIMD3<Float>, to b: SIMD3<Float>) -> Bool {
        let length = simd_length(b - a)
        guard length > 1e-6 else { return !isSolid(a) }
        let steps = Int(ceil(length / (cell * 0.5)))
        for step in 0...steps where isSolid(a + (b - a) * (Float(step) / Float(steps))) {
            return false
        }
        return true
    }
}
