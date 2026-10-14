import ARKit
import AVFoundation
import CoreMotion
import UIKit
import CaptureRules

/// Writes one capture: video.mov (the camera frames, HEVC, portrait), and
/// frames.jsonl with the phone's camera pose and intrinsics for every frame,
/// then capture.json with the recording's report. The pipeline can use the
/// poses and metric scale instead of estimating them.
///
/// For refining those poses on the phone (docs/on-device-plan-2026-10.md) it
/// also keeps the motion sensors' samples between frames (`imu` on each frame),
/// flags a frame whose pose leapt from the last (`jump`), and leaves an anchor
/// at the camera every `anchorEvery` frames. ARKit moves its anchors when it
/// corrects its map, so where they end up (capture.json's keyframeAnchors) are
/// corrected keyframe poses; whether an iPhone without LiDAR moves them at all
/// is what the first recordings will show.
final class Recorder {
    let folder: URL
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var poses: FileHandle?
    private var started = false
    private(set) var framesWritten = 0
    private(set) var framesDropped = 0

    /// An anchor at the camera every this many written frames (twice a second at 30 fps).
    static let anchorEvery = 15
    /// A pose this far from the last frame's is a jump (relocalising), not a hand moving.
    static let jumpMetres: Float = 0.15
    private struct Keyframe {
        let frame: Int
        let placed: simd_float4x4
        var last: simd_float4x4
        var moves = 0
    }
    private var keyframes: [UUID: Keyframe] = [:]
    /// Every keyframe anchor this recording added, so all of them can be taken out of the
    /// session at the end (one added just before Stop is not in the session's frame yet).
    private(set) var placedAnchors: [ARAnchor] = []
    private var lastPosition: SIMD3<Float>?
    /// Every plane ARKit reported while recording, with its raw classification (the room
    /// map writes an unclassified vertical plane as a wall), for capture.json's planeClasses.
    private var planeClasses: [UUID: [String: Any]] = [:]

    private let motion = CMMotionManager()
    private let motionQueue = OperationQueue()
    private let motionLock = NSLock()
    private var motionSamples: [[Double]] = []

    init() throws {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        folder = documents.appendingPathComponent("Captures/Capture-\(formatter.string(from: Date()))")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: folder.appendingPathComponent("frames.jsonl").path, contents: nil)
        poses = try FileHandle(forWritingTo: folder.appendingPathComponent("frames.jsonl"))
        // 100 Hz; the samples carry the same clock as ARFrame.timestamp (seconds since boot).
        motionQueue.maxConcurrentOperationCount = 1
        if motion.isDeviceMotionAvailable {
            motion.deviceMotionUpdateInterval = 1.0 / 100
            motion.startDeviceMotionUpdates(to: motionQueue) { [weak self] data, _ in
                guard let self, let data else { return }
                let g = data.rotationRate, a = data.userAcceleration, down = data.gravity
                let sample = [data.timestamp, g.x, g.y, g.z, a.x + down.x, a.y + down.y, a.z + down.z]
                self.motionLock.lock()
                self.motionSamples.append(sample)
                self.motionLock.unlock()
            }
        }
    }

    /// Appends a frame (called on the capture queue). Returns an anchor for the
    /// caller to add to the session when this frame is a keyframe.
    @discardableResult
    func append(_ frame: ARFrame) -> ARAnchor? {
        let buffer = frame.capturedImage
        if writer == nil { setUpWriter(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer)) }
        guard let writer, let input, let adaptor else { return nil }
        let time = CMTime(seconds: frame.timestamp, preferredTimescale: 600)
        if !started {
            guard writer.startWriting() else { return nil }
            writer.startSession(atSourceTime: time)
            started = true
        }
        guard input.isReadyForMoreMediaData else {
            framesDropped += 1
            return nil
        }
        guard adaptor.append(buffer, withPresentationTime: time) else {
            framesDropped += 1
            return nil
        }
        framesWritten += 1
        writePose(frame)
        guard (framesWritten - 1) % Self.anchorEvery == 0 else { return nil }
        let anchor = ARAnchor(name: "keyframe-\(framesWritten - 1)", transform: frame.camera.transform)
        keyframes[anchor.identifier] = Keyframe(frame: framesWritten - 1, placed: frame.camera.transform,
                                                last: frame.camera.transform)
        placedAnchors.append(anchor)
        return anchor
    }

    /// Counts the times ARKit moved a keyframe anchor, and notes each plane's
    /// classification (called on the capture queue, for added and updated
    /// anchors). An update that moves a keyframe under half a millimetre (and
    /// turns it under about 0.03 degrees) is not a move.
    func anchorsUpdated(_ anchors: [ARAnchor]) {
        for case let plane as ARPlaneAnchor in anchors {
            planeClasses[plane.identifier] = [
                "alignment": plane.alignment == .vertical ? "vertical" : "horizontal",
                "classification": Self.classification(of: plane),
                "extent": [plane.planeExtent.width, plane.planeExtent.height],
                "removed": false,
            ]
        }
        for anchor in anchors {
            guard var keyframe = keyframes[anchor.identifier] else { continue }
            let t = anchor.transform, last = keyframe.last
            let shift = max(simd_distance(t.columns.3, last.columns.3),
                            simd_distance(t.columns.0, last.columns.0), simd_distance(t.columns.2, last.columns.2))
            guard shift > 0.0005 else { continue }
            keyframe.last = t
            keyframe.moves += 1
            keyframes[anchor.identifier] = keyframe
        }
    }

    /// Marks planes ARKit removed (merged into another, usually) so they are not counted twice.
    func anchorsRemoved(_ anchors: [ARAnchor]) {
        for anchor in anchors where planeClasses[anchor.identifier] != nil {
            planeClasses[anchor.identifier]?["removed"] = true
        }
    }

    private static func classification(of plane: ARPlaneAnchor) -> String {
        guard ARPlaneAnchor.isClassificationSupported else { return "unsupported" }
        switch plane.classification {
        case .wall: return "wall"
        case .floor: return "floor"
        case .ceiling: return "ceiling"
        case .table: return "table"
        case .seat: return "seat"
        case .door: return "door"
        case .window: return "window"
        case .none(let status):
            switch status {
            case .notAvailable: return "none (not available)"
            case .undetermined: return "none (undetermined)"
            case .unknown: return "none (unknown)"
            @unknown default: return "none"
            }
        @unknown default: return "other"
        }
    }

    /// Whether an anchor belongs to a recording's keyframes.
    static func isKeyframe(_ anchor: ARAnchor) -> Bool {
        anchor.name?.hasPrefix("keyframe-") == true
    }

    private static func columnMajor(_ t: simd_float4x4) -> [Float] {
        [t.columns.0, t.columns.1, t.columns.2, t.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }

    private func setUpWriter(width: Int, height: Int) {
        let url = folder.appendingPathComponent("video.mov")
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
        let bitrate = width >= 3000 ? 45_000_000 : 18_000_000
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: bitrate],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        // The sensor image is landscape; the phone is held portrait.
        input.transform = CGAffineTransform(rotationAngle: .pi / 2)
        guard writer.canAdd(input) else { return }
        writer.add(input)
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        self.writer = writer
        self.input = input
    }

    private func writePose(_ frame: ARFrame) {
        let camera = frame.camera
        let t = camera.transform
        let k = camera.intrinsics
        let matrix = Self.columnMajor(t)
        var line: [String: Any] = [
            "t": frame.timestamp,
            "frame": framesWritten - 1,
            // Camera to world, column-major; ARKit camera axes: x right, y up, z backward.
            "transform": matrix,
            "intrinsics": [k.columns.0.x, k.columns.1.y, k.columns.2.x, k.columns.2.y],
            "size": [Int(camera.imageResolution.width), Int(camera.imageResolution.height)],
            "tracking": trackingName(camera.trackingState),
            "exposure": camera.exposureDuration,
        ]
        if let light = frame.lightEstimate { line["ambient"] = light.ambientIntensity }
        let position = SIMD3(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        if let last = lastPosition, simd_distance(last, position) > Self.jumpMetres { line["jump"] = true }
        lastPosition = position
        // The motion sensors since the last frame, up to this frame's capture time: [t, gyro x y z
        // (rad/s, bias-corrected), acceleration x y z (g, gravity included)], device axes. Samples
        // taken after it (ARKit hands a frame over some milliseconds after capture) wait for the next.
        motionLock.lock()
        let cut = motionSamples.firstIndex { $0[0] > frame.timestamp } ?? motionSamples.count
        let samples = Array(motionSamples[..<cut])
        motionSamples.removeFirst(cut)
        motionLock.unlock()
        if !samples.isEmpty { line["imu"] = samples }
        // The tracking's points, every tenth frame (about as often as frames are analysed), so that a
        // recording can be played back through the perception on a Mac exactly as the phone had it
        // (tools/phone_capture_export.py). Millimetres; at most 300 of them.
        if (framesWritten - 1) % 10 == 0, let cloud = frame.rawFeaturePoints?.points {
            let step = max(1, cloud.count / 300)
            line["points"] = stride(from: 0, to: cloud.count, by: step).map { i -> [Double] in
                let p = cloud[i]
                return [(Double(p.x) * 1000).rounded() / 1000, (Double(p.y) * 1000).rounded() / 1000, (Double(p.z) * 1000).rounded() / 1000]
            }
        }
        if let data = try? JSONSerialization.data(withJSONObject: line) {
            poses?.write(data)
            poses?.write(Data([0x0A]))
        }
    }

    private func trackingName(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "notAvailable"
        case .limited(let reason):
            switch reason {
            case .initializing: return "initializing"
            case .excessiveMotion: return "excessiveMotion"
            case .insufficientFeatures: return "insufficientFeatures"
            case .relocalizing: return "relocalizing"
            @unknown default: return "limited"
            }
        @unknown default: return "unknown"
        }
    }

    /// Finishes the video and writes capture.json. `anchors` are the session's
    /// anchors at the end. Calls back on the main queue.
    func finish(summary: CaptureSummary, advice: [Rule], config: RuleConfig, format: String,
                objectsSeen: [String: Double], map: RoomMap, anchors: [ARAnchor],
                completion: @escaping (URL?) -> Void) {
        motion.stopDeviceMotionUpdates()
        motionQueue.waitUntilAllOperationsAreFinished()
        try? poses?.close()
        poses = nil
        // Each keyframe's camera where it was placed and where ARKit has it now
        // (camera to world, column-major, as in frames.jsonl), and how often it moved.
        // An anchor added just before Stop is not in the session's last frame yet: its
        // last known pose stands in, and `atStop` says so.
        let finalAnchors = Dictionary(anchors.map { ($0.identifier, $0.transform) }, uniquingKeysWith: { a, _ in a })
        var keyframeAnchors: [[String: Any]] = []
        for (id, keyframe) in keyframes.sorted(by: { $0.value.frame < $1.value.frame }) {
            let now = finalAnchors[id] ?? keyframe.last
            let entry: [String: Any] = ["frame": keyframe.frame, "placed": Self.columnMajor(keyframe.placed),
                                        "final": Self.columnMajor(now), "updates": keyframe.moves,
                                        "atStop": finalAnchors[id] != nil]
            keyframeAnchors.append(entry)
        }
        let planeList = Array(planeClasses.values)
        let write = { [folder, framesWritten, framesDropped] in
            var device = utsname()
            uname(&device)
            let model = withUnsafeBytes(of: &device.machine) { raw in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            let report: [String: Any] = [
                "app": "Oasis Capture 0.1",
                "device": model,
                "system": UIDevice.current.systemVersion,
                "videoFormat": format,
                "framesWritten": framesWritten,
                "framesDropped": framesDropped,
                "keyframeAnchors": keyframeAnchors,
                "planeClasses": planeList,
                "summary": (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(summary))) ?? [:],
                "advice": advice.map { ["rule": $0.rawValue, "message": config.message($0)] },
                // What the on-device segmentation saw, and for how many seconds.
                "objectsSeen": objectsSeen.mapValues { ($0 * 10).rounded() / 10 },
                // Walls, floor and furniture as the phone placed them (world space, metres, y up).
                "roomMap": [
                    "planes": map.planes.map { plane in [
                        "kind": plane.kind.rawValue, "vertical": plane.vertical,
                        "center": [plane.center.x, plane.center.y, plane.center.z],
                        "xAxis": [plane.xAxis.x, plane.xAxis.y, plane.xAxis.z],
                        "zAxis": [plane.zAxis.x, plane.zAxis.y, plane.zAxis.z],
                        "extent": [plane.extent.x, plane.extent.y]] as [String: Any] },
                    "objects": map.objects.map { object in [
                        "label": object.label, "group": object.group, "classId": object.classId,
                        "center": [object.center.x, object.center.y, object.center.z],
                        "size": [object.size.x, object.size.y, object.size.z],
                        "yaw": object.yaw,
                        "voxels": object.points] as [String: Any] },
                ] as [String: Any],
            ]
            if let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: folder.appendingPathComponent("capture.json"))
            }
        }
        guard let writer, let input, started else {
            write()
            DispatchQueue.main.async { completion(nil) }
            return
        }
        input.markAsFinished()
        writer.finishWriting { [folder] in
            write()
            let ok = writer.status == .completed
            DispatchQueue.main.async { completion(ok ? folder : nil) }
        }
    }
}
