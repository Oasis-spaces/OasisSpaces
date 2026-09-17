import ARKit
import SceneKit
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
}

struct CaptureResult: Identifiable {
    let id = UUID()
    let folder: URL?
    let summary: CaptureSummary
    let advice: [Rule]
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

    private let queue = DispatchQueue(label: "capture.frames", qos: .userInteractive)
    private var engine: RuleEngine
    private let analyzer = FrameAnalyzer()
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

    override init() {
        engine = RuleEngine(config: RuleConfig.bundled())
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
            recorder.finish(summary: summary, advice: advice, config: self.config, format: self.formatName) { folder in
                self.state.isFinishing = false
                self.state.result = CaptureResult(folder: folder, summary: summary, advice: advice)
            }
            self.recorder = nil
        }
    }

    // MARK: ARSessionDelegate (on `queue`)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let sample = analyzer.sample(frame)
        let guidance = engine.update(sample)

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
