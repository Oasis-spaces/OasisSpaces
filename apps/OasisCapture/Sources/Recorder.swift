import ARKit
import AVFoundation
import UIKit
import CaptureRules

/// Writes one capture: video.mov (the camera frames, HEVC, portrait), and
/// frames.jsonl with the phone's camera pose and intrinsics for every frame,
/// then capture.json with the recording's report. The pipeline can use the
/// poses and metric scale instead of estimating them.
final class Recorder {
    let folder: URL
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var poses: FileHandle?
    private var started = false
    private(set) var framesWritten = 0
    private(set) var framesDropped = 0

    init() throws {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        folder = documents.appendingPathComponent("Captures/Capture-\(formatter.string(from: Date()))")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: folder.appendingPathComponent("frames.jsonl").path, contents: nil)
        poses = try FileHandle(forWritingTo: folder.appendingPathComponent("frames.jsonl"))
    }

    /// Appends a frame (called on the capture queue).
    func append(_ frame: ARFrame) {
        let buffer = frame.capturedImage
        if writer == nil { setUpWriter(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer)) }
        guard let writer, let input, let adaptor else { return }
        let time = CMTime(seconds: frame.timestamp, preferredTimescale: 600)
        if !started {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: time)
            started = true
        }
        guard input.isReadyForMoreMediaData else {
            framesDropped += 1
            return
        }
        guard adaptor.append(buffer, withPresentationTime: time) else {
            framesDropped += 1
            return
        }
        framesWritten += 1
        writePose(frame)
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
        let matrix = [t.columns.0, t.columns.1, t.columns.2, t.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
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

    /// Finishes the video and writes capture.json. Calls back on the main queue.
    func finish(summary: CaptureSummary, advice: [Rule], config: RuleConfig, format: String,
                objectsSeen: [String: Double], map: RoomMap, completion: @escaping (URL?) -> Void) {
        try? poses?.close()
        poses = nil
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
                        "min": [object.min.x, object.min.y, object.min.z],
                        "max": [object.max.x, object.max.y, object.max.z],
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
