import ARKit
import CoreML
import Vision
import CaptureRules

/// Runs the room segmentation model on camera frames, a few times a second,
/// one frame at a time, off the capture queue. Results are outlines in the
/// camera image's own normalised coordinates (landscape sensor space), ready
/// to map onto the screen with ARFrame.displayTransform.
final class SegmentationRunner {
    let spec = DetectionSpec.bundled()
    /// Seconds between runs.
    var interval: Double = 0.3

    private let request: VNCoreMLRequest?
    private let queue = DispatchQueue(label: "capture.segmentation", qos: .userInitiated)
    private var busy = false
    private var lastRun: Double = -1
    private(set) var latest: SegmentationResult?
    private let lock = NSLock()

    init() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all   // the Neural Engine where there is one
        if let url = Bundle.main.url(forResource: "RoomSegmentation", withExtension: "mlmodelc"),
           let model = try? MLModel(contentsOf: url, configuration: configuration),
           let visionModel = try? VNCoreMLModel(for: model) {
            let request = VNCoreMLRequest(model: visionModel)
            // The whole frame, squeezed to the model's square input: outlines then
            // map back by simple scaling.
            request.imageCropAndScaleOption = .scaleFill
            self.request = request
        } else {
            request = nil
        }
    }

    var isAvailable: Bool { request != nil }

    /// The latest result, safe to read from any queue.
    var current: SegmentationResult? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    /// Starts a run on this frame if the previous one finished and the interval
    /// passed. `done` gets the result on the segmentation queue.
    func submit(_ frame: ARFrame, done: @escaping (SegmentationResult, [SIMD3<Float>], ARCamera) -> Void) {
        guard let request, !busy, frame.timestamp - lastRun >= interval else { return }
        busy = true
        lastRun = frame.timestamp
        let buffer = frame.capturedImage
        // The tracked points of this very frame: labelled once the class map is back.
        let points = frame.rawFeaturePoints?.points ?? []
        let camera = frame.camera
        queue.async { [weak self] in
            guard let self else { return }
            defer { self.busy = false }
            // The sensor image is landscape; .right shows the model the room upright.
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right)
            guard (try? handler.perform([request])) != nil,
                  let observation = request.results?.first as? VNCoreMLFeatureValueObservation,
                  let array = observation.featureValue.multiArrayValue else { return }
            let shape = array.shape.map(\.intValue)
            let height = shape[shape.count - 2], width = shape[shape.count - 1]
            var classes = [Int32](repeating: 0, count: width * height)
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: width * height)
            classes.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: width * height) }
            var result = OutlineExtractor.extract(classes: classes, width: width, height: height, spec: self.spec)
            // Upright (portrait) coordinates back to the sensor image's own:
            // x_sensor = y_upright, y_sensor = 1 - x_upright.
            for i in result.regions.indices {
                result.regions[i].outline = result.regions[i].outline.map { SIMD2($0.y, 1 - $0.x) }
                let c = result.regions[i].centroid
                result.regions[i].centroid = SIMD2(c.y, 1 - c.x)
            }
            self.lock.lock()
            self.latest = result
            self.lock.unlock()
            done(result, points, camera)
        }
    }
}
