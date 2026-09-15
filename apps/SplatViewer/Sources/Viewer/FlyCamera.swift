import AppKit
import simd

/// The keys that move the camera while held.
enum MoveKey: Hashable, CaseIterable {
    case left, right, up, down, w, a, s, d, q, e, pageUp, pageDown

    init?(keyCode: UInt16) {
        switch keyCode {
        case 123: self = .left
        case 124: self = .right
        case 125: self = .down
        case 126: self = .up
        case 13: self = .w
        case 0: self = .a
        case 1: self = .s
        case 2: self = .d
        case 12: self = .q
        case 14: self = .e
        case 116: self = .pageUp
        case 121: self = .pageDown
        default: return nil
        }
    }

    var keyCode: CGKeyCode {
        switch self {
        case .left: 123
        case .right: 124
        case .down: 125
        case .up: 126
        case .w: 13
        case .a: 0
        case .s: 1
        case .d: 2
        case .q: 12
        case .e: 14
        case .pageUp: 116
        case .pageDown: 121
        }
    }

    var isLetter: Bool { [.w, .a, .s, .d, .q, .e].contains(self) }
}

/// Which movement keys are down. macOS does not always deliver a key-up for an arrow
/// released while ⌘ is held, so when ⌘ goes up any key the keyboard no longer reports
/// as pressed is dropped.
struct KeyState {
    private(set) var held: Set<MoveKey> = []

    mutating func press(_ key: MoveKey) { held.insert(key) }
    mutating func release(_ key: MoveKey) { held.remove(key) }
    mutating func releaseAll() { held.removeAll() }

    mutating func commandReleased() {
        held = held.filter { CGEventSource.keyState(.combinedSessionState, key: $0.keyCode) }
    }
}

/// The pipeline's starting camera, written beside a splat as <name>.view.json
/// (pipeline/splat_export.py): a COLMAP-style world-to-camera matrix, column-major.
struct StartView: Decodable {
    var viewMatrix: [Double]
    var fovY: Double?
    var up: [Double]?
    var metre: Double?
    var frame: String?

    static func load(besides url: URL) -> StartView? {
        let file = url.deletingPathExtension().appendingPathExtension("view.json")
        guard let data = try? Data(contentsOf: file),
              let view = try? JSONDecoder().decode(StartView.self, from: data),
              view.viewMatrix.count == 16 else { return nil }
        return view
    }
}

struct SceneBounds {
    var center: SIMD3<Float>
    /// Length of the diagonal between the 5th and 95th percentile corners.
    var extent: Float

    init(positions: [SIMD3<Float>]) {
        guard !positions.isEmpty else {
            center = .zero
            extent = 1
            return
        }
        var xs = positions.map(\.x), ys = positions.map(\.y), zs = positions.map(\.z)
        xs.sort(); ys.sort(); zs.sort()
        let low = Int(Float(positions.count - 1) * 0.05), high = Int(Float(positions.count - 1) * 0.95)
        let lo = SIMD3(xs[low], ys[low], zs[low]), hi = SIMD3(xs[high], ys[high], zs[high])
        center = (lo + hi) / 2
        extent = max(simd_length(hi - lo), 1e-3)
    }
}

/// A walking camera: position, a turn (yaw) around the scene's up direction and a tilt
/// (pitch) above or below level. Movement stays level, so a room is walked, not flown.
struct FlyCamera {
    struct Pose {
        var position: SIMD3<Float>
        var yaw: Float
        var pitch: Float
    }

    /// What the held keys ask for this frame.
    struct Input {
        var move = SIMD3<Float>.zero   // forward, right, up
        var turn = SIMD2<Float>.zero   // yaw (left +), pitch (up +)
        var fast = false

        init() {}

        /// Arrows move; ⌘ + arrows turn and look; ⌥ + ↑↓ move up and down; ⇧ goes faster.
        init(held: Set<MoveKey>, flags: NSEvent.ModifierFlags) {
            let command = flags.contains(.command)
            let option = flags.contains(.option)
            fast = flags.contains(.shift)
            for key in held {
                switch key {
                case .up:
                    if command { turn.y += 1 } else if option { move.z += 1 } else { move.x += 1 }
                case .down:
                    if command { turn.y -= 1 } else if option { move.z -= 1 } else { move.x -= 1 }
                case .left:
                    if command { turn.x += 1 } else { move.y -= 1 }
                case .right:
                    if command { turn.x -= 1 } else { move.y += 1 }
                case .w: move.x += 1
                case .s: move.x -= 1
                case .a: move.y -= 1
                case .d: move.y += 1
                case .e, .pageUp: move.z += 1
                case .q, .pageDown: move.z -= 1
                }
            }
        }
    }

    static let maxPitch: Float = 85 * .pi / 180
    static let turnRate: Float = 75 * .pi / 180

    private(set) var up = SIMD3<Float>(0, -1, 0)
    private(set) var forward0 = SIMD3<Float>(0, 0, 1)
    var pose = Pose(position: .zero, yaw: 0, pitch: 0)
    private(set) var startPose = Pose(position: .zero, yaw: 0, pitch: 0)
    private(set) var fovY: Float = 55 * .pi / 180
    /// Scene units per second.
    private(set) var walkSpeed: Float = 1
    private(set) var near: Float = 0.01
    private(set) var far: Float = 1000
    private var velocity = SIMD3<Float>.zero
    private var turnVelocity = SIMD2<Float>.zero

    var isMoving: Bool {
        simd_length(velocity) > walkSpeed * 0.005 || simd_length(turnVelocity) > 0.005
    }

    /// Starts at the pipeline's starting camera when there is one. Otherwise the scene is
    /// taken to be a 3D Gaussian splatting capture (y down, cameras looking along +z) and
    /// viewed from in front of its middle.
    mutating func configure(bounds: SceneBounds, start: StartView?) {
        let extent = bounds.extent
        walkSpeed = extent * 0.12
        near = extent * 0.0008
        if let start {
            let m = start.viewMatrix.map { Float($0) }
            let right = SIMD3(m[0], m[4], m[8])
            let down = SIMD3(m[1], m[5], m[9])
            let forward = SIMD3(m[2], m[6], m[10])
            let t = SIMD3(m[12], m[13], m[14])
            let position = -(right * t.x + down * t.y + forward * t.z)
            var upVector = -down
            if let u = start.up, u.count == 3 {
                upVector = SIMD3(Float(u[0]), Float(u[1]), Float(u[2]))
            }
            setBasis(up: upVector, forward: forward)
            let pitch = asin(min(max(simd_dot(simd_normalize(forward), up), -1), 1))
            startPose = Pose(position: position, yaw: 0, pitch: Self.clampPitch(pitch))
            if let fov = start.fovY { fovY = Float(fov) * .pi / 180 }
            if let metre = start.metre, metre > 0 {
                walkSpeed = Float(metre) * 1.2     // about walking pace
                near = Float(metre) * 0.02
            }
        } else {
            setBasis(up: SIMD3(0, -1, 0), forward: SIMD3(0, 0, 1))
            startPose = Pose(position: bounds.center - forward0 * extent * 0.75, yaw: 0, pitch: 0)
        }
        far = max(extent * 8, near * 2000)
        reset()
    }

    private mutating func setBasis(up newUp: SIMD3<Float>, forward: SIMD3<Float>) {
        up = simd_normalize(newUp)
        var level = forward - up * simd_dot(forward, up)
        if simd_length(level) < 1e-4 {
            let helper: SIMD3<Float> = abs(up.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 0, 1)
            level = helper - up * simd_dot(helper, up)
        }
        forward0 = simd_normalize(level)
    }

    /// The level direction the camera faces.
    func heading(_ pose: Pose) -> SIMD3<Float> {
        cos(pose.yaw) * forward0 + sin(pose.yaw) * simd_cross(up, forward0)
    }

    func lookDirection(_ pose: Pose) -> SIMD3<Float> {
        cos(pose.pitch) * heading(pose) + sin(pose.pitch) * up
    }

    mutating func advance(dt: Float, input: Input) {
        var move = input.move
        if simd_length(move) > 1 { move = simd_normalize(move) }
        let speed = walkSpeed * (input.fast ? 3 : 1)
        velocity += (move * speed - velocity) * (1 - exp(-dt * 10))
        let rate = Self.turnRate * (input.fast ? 1.8 : 1)
        turnVelocity += (input.turn * rate - turnVelocity) * (1 - exp(-dt * 14))
        let heading = heading(pose)
        let right = simd_cross(heading, up)
        pose.position += (heading * velocity.x + right * velocity.y + up * velocity.z) * dt
        pose.yaw += turnVelocity.x * dt
        pose.pitch = Self.clampPitch(pose.pitch + turnVelocity.y * dt)
    }

    /// Mouse look, in radians.
    mutating func look(yaw: Float, pitch: Float) {
        pose.yaw += yaw
        pose.pitch = Self.clampPitch(pose.pitch + pitch)
    }

    /// A step from a scroll or pinch, in seconds of walking.
    mutating func step(forward: Float, right: Float) {
        let heading = heading(pose)
        pose.position += (heading * forward + simd_cross(heading, up) * right) * walkSpeed
    }

    mutating func reset() {
        pose = startPose
        velocity = .zero
        turnVelocity = .zero
    }

    /// World-to-camera matrix, the camera looking along -z with y up.
    func viewMatrix(_ pose: Pose? = nil) -> simd_float4x4 {
        let pose = pose ?? self.pose
        let f = lookDirection(pose)
        let r = simd_normalize(simd_cross(f, up))
        let u = simd_cross(r, f)
        let p = pose.position
        return simd_float4x4(columns: (
            SIMD4(r.x, u.x, -f.x, 0),
            SIMD4(r.y, u.y, -f.y, 0),
            SIMD4(r.z, u.z, -f.z, 0),
            SIMD4(-simd_dot(r, p), -simd_dot(u, p), simd_dot(f, p), 1)))
    }

    /// Right-handed perspective with Metal's 0...1 depth. A larger `near` hides everything
    /// closer than it: how the viewer looks past a wall it has backed into.
    func projection(aspect: Float, near nearOverride: Float? = nil) -> simd_float4x4 {
        let near = max(nearOverride ?? self.near, self.near)
        let ys = 1 / tan(fovY / 2)
        let xs = ys / aspect
        let zs = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4(xs, 0, 0, 0),
            SIMD4(0, ys, 0, 0),
            SIMD4(0, 0, zs, -1),
            SIMD4(0, 0, zs * near, 0)))
    }

    /// The ray through a point of the view (points, origin top-left), in the splat's frame.
    func ray(through point: CGPoint, in size: CGSize) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let ndc = SIMD2(Float(2 * point.x / max(size.width, 1) - 1), Float(1 - 2 * point.y / max(size.height, 1)))
        let tanY = tan(fovY / 2)
        let aspect = Float(size.width / max(size.height, 1))
        let camera = SIMD3(ndc.x * tanY * aspect, ndc.y * tanY, -1)
        let inverse = viewMatrix().inverse
        let direction = simd_normalize(SIMD3((inverse * SIMD4(camera, 0)).x, (inverse * SIMD4(camera, 0)).y,
                                             (inverse * SIMD4(camera, 0)).z))
        return (pose.position, direction)
    }

    /// Where a splat-frame point lands in the view (points, origin top-left); nil when
    /// it is behind the camera.
    func project(_ point: SIMD3<Float>, in size: CGSize) -> CGPoint? {
        let aspect = Float(size.width / max(size.height, 1))
        let clip = projection(aspect: aspect) * viewMatrix() * SIMD4(point, 1)
        guard clip.w > near * 0.5 else { return nil }
        return CGPoint(x: CGFloat((clip.x / clip.w + 1) / 2) * size.width,
                       y: CGFloat((1 - clip.y / clip.w) / 2) * size.height)
    }

    private static func clampPitch(_ pitch: Float) -> Float {
        min(max(pitch, -maxPitch), maxPitch)
    }
}
