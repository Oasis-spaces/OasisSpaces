import ARKit
import SceneKit
import SwiftUI
import UIKit
import CaptureRules

/// What the capture screen shows. Updated on the main queue only.
final class CaptureState: ObservableObject {
    @Published var guidance: Guidance?
    @Published var isRecording = false
    @Published var isFinishing = false
    @Published var elapsed: Double = 0
    @Published var coverage: Double = 0
    @Published var sectors: Set<Int> = []
    @Published var floorSeen = false
    @Published var heading: Double = 0
    @Published var position = SIMD2<Float>(0, 0)
    @Published var path: [SIMD2<Float>] = []
    @Published var mapPoints: [SIMD2<Float>] = []
    @Published var pointCount = 0
    @Published var format = ""
    @Published var result: CaptureResult?
    @Published var regions: [Region] = []
    @Published var showOutlines = true
    @Published var detected: [String] = []
    @Published var map = RoomMap()
}

struct CaptureResult: Identifiable {
    let id = UUID()
    let folder: URL?
    let summary: CaptureSummary
    let advice: [Rule]
    /// Recognised classes, most seen first.
    var objectsSeen: [String] = []
    var map = RoomMap()
}

/// Runs the AR session: every frame goes through the rules (before and during
/// recording), into the live scan, and while recording into the video and
/// pose files. Frame work happens on `queue`; the screen state on main.
final class CaptureController: NSObject, ARSessionDelegate {
    let session = ARSession()
    let state = CaptureState()
    let config = RuleConfig.bundled()
    /// The scan's points, drawn in the camera view.
    let cloudNode = SCNNode()
    /// Detected planes and furniture boxes, drawn in the camera view.
    let mapNode = SCNNode()
    let mapBuilder: RoomMapBuilder

    private let queue = DispatchQueue(label: "capture.frames", qos: .userInteractive)
    private var engine: RuleEngine
    private let analyzer = FrameAnalyzer()
    let segmentation = SegmentationRunner()
    /// Seconds each detected class was in view while recording (for capture.json).
    private var secondsSeen: [String: Double] = [:]
    private var lastSegmentationTime: Double?
    private let cloud = ScanCloud()
    private var recorder: Recorder?
    private var recording = false
    private var path: [SIMD2<Float>] = []
    private var lastPublish: Double = 0
    private var lastCloudUpdate: Double = 0
    private var lastHaptic: Double = 0
    private var shownRule: Rule?
    private var formatName = ""
    private let haptics = UINotificationFeedbackGenerator()

    private var planeNodes: [UUID: SCNNode] = [:]
    private var lastMapBuild: Double = 0
    private var latestMap = RoomMap()

    override init() {
        engine = RuleEngine(config: RuleConfig.bundled())
        mapBuilder = RoomMapBuilder(spec: DetectionSpec.bundled())
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    func start() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        configuration.isLightEstimationEnabled = true
        configuration.planeDetection = [.horizontal, .vertical]
        // 4K frames where the phone offers them (splats from 4K frames are sharper).
        if let format = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution {
            configuration.videoFormat = format
        }
        let f = configuration.videoFormat
        formatName = "\(Int(f.imageResolution.width))x\(Int(f.imageResolution.height)) @ \(f.framesPerSecond) fps"
        DispatchQueue.main.async { self.state.format = self.formatName }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        UIApplication.shared.isIdleTimerDisabled = true
        haptics.prepare()
    }

    func pause() {
        session.pause()
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func startRecording() {
        queue.async {
            guard !self.recording else { return }
            do {
                self.recorder = try Recorder()
            } catch {
                return
            }
            self.engine.startRecording()
            self.mapBuilder.reset()
            for (id, node) in self.planeNodes { _ = id; DispatchQueue.main.async { node.removeFromParentNode() } }
            self.planeNodes = [:]
            self.secondsSeen = [:]
            self.lastSegmentationTime = nil
            self.cloud.reset()
            self.path = []
            self.recording = true
            DispatchQueue.main.async {
                self.state.isRecording = true
                self.state.result = nil
            }
        }
    }

    func stopRecording() {
        queue.async {
            guard self.recording, let recorder = self.recorder else { return }
            self.recording = false
            let (summary, advice) = self.engine.finish()
            DispatchQueue.main.async {
                self.state.isRecording = false
                self.state.isFinishing = true
            }
            recorder.finish(summary: summary, advice: advice, config: self.config, format: self.formatName,
                            objectsSeen: self.secondsSeen, map: self.mapBuilder.build()) { folder in
                self.state.isFinishing = false
                self.state.result = CaptureResult(folder: folder, summary: summary, advice: advice,
                                                  objectsSeen: self.state.detected, map: self.latestMap)
            }
            self.recorder = nil
        }
    }

    // MARK: ARSessionDelegate (on `queue`)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        var sample = analyzer.sample(frame)
        if let result = segmentation.current {
            // People from the segmentation, in place of a separate detector.
            sample.peopleInView = result.personShare >= segmentation.spec.personWarnShare ? 1 : 0
        }
        let guidance = engine.update(sample)
        segmentation.submit(frame) { [weak self] result, points, camera in
            self?.segmented(result, at: frame.timestamp, points: points, camera: camera)
        }

        if recording {
            recorder?.append(frame)
            if frame.camera.trackingState == .normal {
                path.append(SIMD2(sample.position.x, sample.position.z))
                addScanPoints(frame)
            }
        }

        if let guidance, guidance.isNew, guidance.severity >= .warn,
           frame.timestamp - lastHaptic > config.hapticGapSeconds {
            lastHaptic = frame.timestamp
            let kind: UINotificationFeedbackGenerator.FeedbackType = guidance.severity == .stop ? .error : .warning
            DispatchQueue.main.async { self.haptics.notificationOccurred(kind) }
        }

        // The banner changes at once; the rest of the screen ten times a second.
        let changed = guidance?.rule != shownRule
        shownRule = guidance?.rule
        if changed || frame.timestamp - lastPublish > 0.1 {
            lastPublish = frame.timestamp
            publish(sample: sample, guidance: guidance)
        }
        if recording, frame.timestamp - lastCloudUpdate > 0.5 {
            lastCloudUpdate = frame.timestamp
            updateCloudNode()
        }
        if frame.timestamp - lastMapBuild > 1.0 {
            lastMapBuild = frame.timestamp
            rebuildMap()
        }
    }

    // MARK: Planes (walls, floor) from the tracking

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for case let plane as ARPlaneAnchor in anchors { planeChanged(plane) }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for case let plane as ARPlaneAnchor in anchors { planeChanged(plane) }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for case let plane as ARPlaneAnchor in anchors {
            mapBuilder.remove(plane: plane.identifier)
            let node = planeNodes.removeValue(forKey: plane.identifier)
            DispatchQueue.main.async { node?.removeFromParentNode() }
        }
    }

    private func planeChanged(_ anchor: ARPlaneAnchor) {
        let t = anchor.transform
        let rotation = simd_float3x3(SIMD3(t.columns.0.x, t.columns.0.y, t.columns.0.z),
                                     SIMD3(t.columns.1.x, t.columns.1.y, t.columns.1.z),
                                     SIMD3(t.columns.2.x, t.columns.2.y, t.columns.2.z))
        let extent = anchor.planeExtent
        // The extent is turned about the anchor's y by rotationOnYAxis.
        let turn = simd_quatf(angle: extent.rotationOnYAxis, axis: SIMD3(0, 1, 0))
        let xAxis = rotation * turn.act(SIMD3(1, 0, 0))
        let zAxis = rotation * turn.act(SIMD3(0, 0, 1))
        let c = t * SIMD4(anchor.center, 1)
        let info = PlaneInfo(id: anchor.identifier, kind: Self.kind(of: anchor), vertical: anchor.alignment == .vertical,
                             center: SIMD3(c.x, c.y, c.z), xAxis: simd_normalize(xAxis), zAxis: simd_normalize(zAxis),
                             extent: SIMD2(extent.width, extent.height))
        mapBuilder.update(plane: info)
        updatePlaneNode(anchor, kind: info.kind)
    }

    private static func kind(of anchor: ARPlaneAnchor) -> PlaneInfo.Kind {
        if ARPlaneAnchor.isClassificationSupported {
            switch anchor.classification {
            case .wall: return .wall
            case .floor: return .floor
            case .ceiling: return .ceiling
            case .table: return .table
            case .seat: return .seat
            case .door: return .door
            case .window: return .window
            case .none(_): break
            @unknown default: break
            }
        }
        return anchor.alignment == .vertical ? .wall : .floor
    }

    /// A translucent surface with an outline, in the plane's own shape.
    private func updatePlaneNode(_ anchor: ARPlaneAnchor, kind: PlaneInfo.Kind) {
        let color = Self.planeColor(kind)
        DispatchQueue.main.async {
            let node: SCNNode
            if let existing = self.planeNodes[anchor.identifier] {
                node = existing
            } else {
                node = SCNNode()
                guard let device = MTLCreateSystemDefaultDevice(),
                      let geometry = ARSCNPlaneGeometry(device: device) else { return }
                let fill = SCNMaterial()
                fill.diffuse.contents = color.withAlphaComponent(kind == .floor ? 0.10 : 0.16)
                fill.lightingModel = .constant
                fill.isDoubleSided = true
                geometry.materials = [fill]
                node.geometry = geometry
                let outline = SCNNode()
                let edges = ARSCNPlaneGeometry(device: device)!
                let line = SCNMaterial()
                line.diffuse.contents = color
                line.fillMode = .lines
                line.lightingModel = .constant
                line.isDoubleSided = true
                edges.materials = [line]
                outline.geometry = edges
                outline.name = "outline"
                node.addChildNode(outline)
                self.mapNode.addChildNode(node)
                self.planeNodes[anchor.identifier] = node
            }
            node.simdTransform = anchor.transform
            (node.geometry as? ARSCNPlaneGeometry)?.update(from: anchor.geometry)
            (node.childNode(withName: "outline", recursively: false)?.geometry as? ARSCNPlaneGeometry)?.update(from: anchor.geometry)
        }
    }

    private static func planeColor(_ kind: PlaneInfo.Kind) -> UIColor {
        switch kind {
        case .floor: return UIColor(red: 0.56, green: 0.64, blue: 0.72, alpha: 1)
        case .wall, .ceiling: return UIColor(red: 0.56, green: 0.64, blue: 0.72, alpha: 1)
        case .door: return .systemGreen
        case .window: return .systemCyan
        case .table, .seat: return .systemOrange
        case .unknown: return .white
        }
    }

    /// Furniture boxes as wireframes; the map for the screen.
    private func rebuildMap() {
        let map = mapBuilder.build()
        guard map != latestMap else { return }
        latestMap = map
        let spec = mapBuilder.spec
        DispatchQueue.main.async {
            self.state.map = map
            for child in self.mapNode.childNodes where child.name == "box" { child.removeFromParentNode() }
            for object in map.objects {
                let size = object.size
                let box = SCNBox(width: CGFloat(size.x), height: CGFloat(size.y), length: CGFloat(size.z), chamferRadius: 0)
                let material = SCNMaterial()
                material.diffuse.contents = UIColor(Color(hex: spec.groups[object.group]?.color ?? "#FFFFFF"))
                material.fillMode = .lines
                material.lightingModel = .constant
                material.isDoubleSided = true
                box.materials = [material]
                let node = SCNNode(geometry: box)
                node.name = "box"
                node.simdPosition = object.center
                self.mapNode.addChildNode(node)
            }
        }
    }

    /// A segmentation finished (on the segmentation queue): label the frame's
    /// tracked points with it for the room map, and count what was seen.
    private func segmented(_ result: SegmentationResult, at time: Double, points: [SIMD3<Float>], camera: ARCamera) {
        let size = camera.imageResolution
        var labelled: [(SIMD3<Float>, Int)] = []
        labelled.reserveCapacity(points.count)
        for p in points {
            let q = camera.projectPoint(p, orientation: .landscapeRight, viewportSize: size)
            guard q.x >= 0, q.y >= 0, q.x < size.width, q.y < size.height else { continue }
            // Sensor (landscape) coordinates to the upright class map: the model saw the image rotated.
            let xs = Double(q.x / size.width), ys = Double(q.y / size.height)
            if let cls = result.classAt(x: 1 - ys, y: xs) { labelled.append((p, cls)) }
        }
        mapBuilder.add(points: labelled)
        queue.async {
            if self.recording {
                let dt = min(1, time - (self.lastSegmentationTime ?? time))
                for (id, share) in result.classShares where share >= self.segmentation.spec.minRegionShare {
                    if let info = self.segmentation.spec.info(id), info.outline {
                        self.secondsSeen[info.label, default: 0] += dt
                    }
                }
            }
            self.lastSegmentationTime = time
            let detected = self.secondsSeen.sorted { $0.value > $1.value }.map(\.key)
            DispatchQueue.main.async {
                self.state.regions = result.regions
                self.state.detected = detected
            }
        }
    }

    /// New tracked points of this frame, coloured from the image.
    private func addScanPoints(_ frame: ARFrame) {
        guard let points = frame.rawFeaturePoints?.points else { return }
        for p in points.prefix(400) {
            if let color = sampleColor(of: p, frame: frame) {
                cloud.add(p, color: color)
            }
        }
    }

    private func publish(sample: FrameSample, guidance: Guidance?) {
        let coverage = engine.coverage
        let sectors = engine.sectors
        let elapsed = recording ? engine.elapsed : 0
        let floorSeen = recording && engine.floorSeen
        let path = self.path.count > 600 ? stride(from: 0, to: self.path.count, by: self.path.count / 600).map { self.path[$0] } : self.path
        let positions = cloud.positions
        let step = max(1, positions.count / 2500)
        let mapPoints = stride(from: 0, to: positions.count, by: step).map { SIMD2(positions[$0].x, positions[$0].z) }
        let count = positions.count
        DispatchQueue.main.async {
            let s = self.state
            if s.guidance != guidance { s.guidance = guidance }
            s.elapsed = elapsed
            s.coverage = coverage
            s.sectors = sectors
            s.floorSeen = floorSeen
            s.heading = sample.heading
            s.position = SIMD2(sample.position.x, sample.position.z)
            s.path = path
            s.mapPoints = mapPoints
            s.pointCount = count
        }
    }

    /// Rebuilds the point geometry drawn over the camera view.
    private func updateCloudNode() {
        let positions = cloud.positions
        let colors = cloud.colors
        guard !positions.isEmpty else { return }
        let vertices = positions.map { SCNVector3($0.x, $0.y, $0.z) }
        let vertexSource = SCNGeometrySource(vertices: vertices)
        var rgb = [Float]()
        rgb.reserveCapacity(colors.count * 3)
        for c in colors { rgb.append(contentsOf: [c.x, c.y, c.z]) }
        let colorData = rgb.withUnsafeBufferPointer { Data(buffer: $0) }
        let colorSource = SCNGeometrySource(data: colorData, semantic: .color, vectorCount: colors.count,
                                            usesFloatComponents: true, componentsPerVector: 3,
                                            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
                                            dataStride: MemoryLayout<Float>.size * 3)
        let indices = (0..<Int32(positions.count)).map { $0 }
        let element = SCNGeometryElement(indices: indices, primitiveType: .point)
        element.pointSize = 5
        element.minimumPointScreenSpaceRadius = 2
        element.maximumPointScreenSpaceRadius = 6
        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        geometry.materials = [material]
        DispatchQueue.main.async { self.cloudNode.geometry = geometry }
    }
}
