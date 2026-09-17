import simd

/// The live scan: tracked feature points in world space, one per 4 cm voxel,
/// coloured from the camera image when they were first seen.
final class ScanCloud {
    private(set) var positions: [SIMD3<Float>] = []
    private(set) var colors: [SIMD3<Float>] = []
    private var voxels = Set<SIMD3<Int32>>()
    private let voxelSize: Float = 0.04
    let maxPoints = 60_000

    var count: Int { positions.count }

    /// Adds a point unless its voxel already has one. Returns whether it was new.
    @discardableResult
    func add(_ point: SIMD3<Float>, color: SIMD3<Float>) -> Bool {
        guard positions.count < maxPoints else { return false }
        let key = SIMD3<Int32>(Int32((point.x / voxelSize).rounded(.down)),
                               Int32((point.y / voxelSize).rounded(.down)),
                               Int32((point.z / voxelSize).rounded(.down)))
        guard voxels.insert(key).inserted else { return false }
        positions.append(point)
        colors.append(color)
        return true
    }

    func reset() {
        positions.removeAll(keepingCapacity: true)
        colors.removeAll(keepingCapacity: true)
        voxels.removeAll(keepingCapacity: true)
    }
}
