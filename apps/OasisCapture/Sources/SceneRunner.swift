import ARKit
import CoreML
import Vision
import CaptureRules

/// One detected thing in a frame, ready for the screen and the room.
struct InstanceRegion {
    var classIndex: Int
    var confidence: Float
    /// Upright, normalised image coordinates.
    var outline: [SIMD2<Double>]
    var centroid: SIMD2<Double>
    var share: Double
    /// World points behind the mask (empty without depth).
    var points: [SIMD3<Float>]
    /// The outline and centre in the room, for drawing through the live camera.
    var worldOutline: [SIMD3<Float>]
    var worldCentroid: SIMD3<Float>
}

/// What one analysed frame tells us.
struct FrameUnderstanding {
    var time: Double
    /// Upright (portrait) regions and class map of the surfaces: walls, floor, ceiling, doors, windows.
    var segmentation: SegmentationResult
    /// How the depth model's output was scaled to metres, if it could be.
    var depthFit: DepthScale.Fit?
    /// The things the detector found, largest first.
    var instances: [InstanceRegion]
    /// The camera that took the frame.
    var camera: PinholeCamera
}

/// Runs the three models on camera frames a few times a second, one frame at
/// a time, off the capture queue:
///
/// - the object detector (YOLOE prompted with the room vocabulary) gives one
///   mask per thing: this sofa, that wardrobe, each pillow;
/// - the surface segmentation (SegFormer on ADE20K) gives the walls, floor,
///   ceiling, doors and windows, which a detector does not do well;
/// - the depth model (Depth Anything V2) gives a value for every pixel
///   (relative, not metres); the tracking's feature points give the true
///   depth at a few dozen pixels, so a scale is fitted per frame and every
///   pixel of a detected thing becomes a 3D point. From any angle those points
///   land on the same object in the room, so it fuses into one box instead of
///   a new one per viewpoint.
final class SceneRunner {
    /// One runner for the app: its models are loaded once, starting at launch.
    static let shared = SceneRunner()

    let spec = DetectionSpec.bundled()
    let objects = ObjectSpec.bundled()
    /// Seconds between runs.
    var interval: Double = 0.25
    /// Most world points sampled per detected thing.
    var pointsPerInstance = 500

    private var segmentation: VNCoreMLRequest?
    private var depth: VNCoreMLRequest?
    private var detector: VNCoreMLRequest?
    private let queue = DispatchQueue(label: "capture.scene", qos: .userInitiated)
    private var busy = false
    private var lastRun: Double = -1
    private var latest: SegmentationResult?
    private let lock = NSLock()
    private var loading = false
    private(set) var loadSeconds: Double = 0
    /// Rolling average of the time one frame's analysis takes, seconds.
    private(set) var analysisSeconds: Double = 0
    private var runs = 0

    /// Loads the models off the main thread. The first load of a model on a
    /// phone compiles it for the Neural Engine, which can take tens of
    /// seconds; done on the main thread, iOS kills the app for hanging.
    func preload() {
        lock.lock()
        let start = !loading && segmentation == nil
        loading = true
        lock.unlock()
        guard start else { return }
        queue.async { [self] in
            let began = Date()
            /// The Neural Engine first, where the model compiles for it; the GPU
            /// otherwise (the Neural Engine compiler crashes on some layers, and a
            /// model it cannot take still runs fine on the GPU, just slower).
            func request(_ name: String) -> VNCoreMLRequest? {
                guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                    AppLog.write("\(name) is not in the app bundle")
                    return nil
                }
                for units in [MLComputeUnits.all, .cpuAndGPU] {
                    let configuration = MLModelConfiguration()
                    configuration.computeUnits = units
                    let started = Date()
                    do {
                        let model = try MLModel(contentsOf: url, configuration: configuration)
                        let request = VNCoreMLRequest(model: try VNCoreMLModel(for: model))
                        // The whole frame, squeezed to the model's input: results map back by scaling.
                        request.imageCropAndScaleOption = .scaleFill
                        AppLog.write(String(format: "%@ loaded in %.1f s (%@)", name, Date().timeIntervalSince(started),
                                     units == .all ? "Neural Engine" : "GPU"))
                        return request
                    } catch {
                        AppLog.write("\(name) failed to load (\(units == .all ? "Neural Engine" : "GPU")): \(error)")
                    }
                }
                return nil
            }
            let segmentation = request("RoomSegmentation")
            let depth = request("RoomDepth")
            let detector = request("RoomObjects")
            self.lock.lock()
            self.segmentation = segmentation
            self.depth = depth
            self.detector = detector
            self.loadSeconds = Date().timeIntervalSince(began)
            self.lock.unlock()
            AppLog.write(String(format: "models ready in %.1f s (surfaces %@, depth %@, objects %@)",
                         self.loadSeconds, segmentation == nil ? "missing" : "ok", depth == nil ? "missing" : "ok",
                         detector == nil ? "missing" : "ok"))
        }
    }

    var isReady: Bool { lock.withLock { segmentation != nil } }
    var hasDepth: Bool { lock.withLock { depth != nil } }

    /// The latest surface segmentation, safe to read from any queue.
    var current: SegmentationResult? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    /// Analyses this frame if the previous run finished and the interval has
    /// passed. `done` gets the result on the scene queue.
    func submit(_ frame: ARFrame, done: @escaping (FrameUnderstanding) -> Void) {
        let (segmentation, depthRequest, detector) = lock.withLock { (self.segmentation, self.depth, self.detector) }
        guard let segmentation, !busy, frame.timestamp - lastRun >= interval else { return }
        busy = true
        lastRun = frame.timestamp
        let buffer = frame.capturedImage
        let points = frame.rawFeaturePoints?.points ?? []
        let camera = frame.camera
        let time = frame.timestamp
        queue.async { [weak self] in
            guard let self else { return }
            defer { self.busy = false }
            let began = Date()

            // 1. Surfaces, on the upright image.
            let upright = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right)
            guard (try? upright.perform([segmentation])) != nil,
                  let observation = segmentation.results?.first as? VNCoreMLFeatureValueObservation,
                  let array = observation.featureValue.multiArrayValue else { return }
            let shape = array.shape.map(\.intValue)
            let height = shape[shape.count - 2], width = shape[shape.count - 1]
            var classes = [Int32](repeating: 0, count: width * height)
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: width * height)
            classes.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: width * height) }
            let result = OutlineExtractor.extract(classes: classes, width: width, height: height, spec: self.spec)
            self.lock.lock()
            self.latest = result
            self.lock.unlock()

            // 2. Things, on the upright image.
            var instances: [Instance] = []
            if let detector, (try? upright.perform([detector])) != nil {
                instances = self.decode(detector)
            }

            // 3. Depth, on the sensor image as it is (landscape, like the model was trained).
            let pinhole = Self.pinhole(camera)
            var understanding = FrameUnderstanding(time: time, segmentation: result, depthFit: nil,
                                                   instances: [], camera: pinhole)
            var depthValues: DepthValues?
            if let depthRequest,
               (try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([depthRequest])) != nil,
               let depthMap = (depthRequest.results?.first as? VNPixelBufferObservation)?.pixelBuffer,
               let values = Self.floats(from: depthMap) {
                let fit = Self.fitScale(values, pinhole: pinhole, points: points)
                understanding.depthFit = fit
                if let fit, fit.error < 0.2 { depthValues = values.scaled(fit) }
            }
            understanding.instances = instances.map { self.region($0, depth: depthValues, pinhole: pinhole) }
            // The surfaces' outlines go into the room too, so they follow the camera like the things do.
            for i in understanding.segmentation.regions.indices {
                let region = understanding.segmentation.regions[i]
                let lifted = OutlineLift.lift(outline: region.outline, centroid: region.centroid, inset: 0.03,
                                              camera: pinhole) { x, y in depthValues?.metres(x: x, y: y) }
                understanding.segmentation.regions[i].worldOutline = lifted.outline
                understanding.segmentation.regions[i].worldCentroid = lifted.centroid
            }
            let took = Date().timeIntervalSince(began)
            self.analysisSeconds = self.analysisSeconds == 0 ? took : self.analysisSeconds * 0.9 + took * 0.1
            self.runs += 1
            if self.runs % 20 == 0 {
                AppLog.write(String(format: "analysis %.0f ms/frame, %d things, depth %@", self.analysisSeconds * 1000,
                             instances.count, understanding.depthFit.map { String(format: "fit %.2f", $0.error) } ?? "none"))
            }
            done(understanding)
        }
    }

    // MARK: The detector's outputs

    /// The detector's two outputs (boxes with class scores per anchor; mask
    /// prototypes) into instances.
    private func decode(_ request: VNCoreMLRequest) -> [Instance] {
        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else { return [] }
        var predictions: MLMultiArray?, protos: MLMultiArray?
        for r in results {
            guard let array = r.featureValue.multiArrayValue else { continue }
            if array.shape.count == 4 && array.shape[1].intValue == 32 { protos = array }
            else if array.shape.count == 3 { predictions = array }
        }
        guard let predictions, let protos else { return [] }
        let anchors = predictions.shape[2].intValue
        let channels = predictions.shape[1].intValue
        guard channels == 4 + objects.classes.count + 32 else {
            AppLog.write("the object model has \(channels - 36) classes, object-classes.json \(objects.classes.count)")
            return []
        }
        let maskHeight = protos.shape[2].intValue, maskWidth = protos.shape[3].intValue
        let p = Self.floats(predictions), q = Self.floats(protos)
        return InstanceDecoder.decode(predictions: p, anchors: anchors, protos: q, maskWidth: maskWidth,
                                      maskHeight: maskHeight, spec: objects)
    }

    /// A multi-array's values as floats, whatever it holds.
    private static func floats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            let p = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: p, count: count) }
        case .float16:
            let p = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        case .double:
            let p = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        default:
            for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }

    /// An instance as an outline for the screen and world points for the room.
    private func region(_ instance: Instance, depth: DepthValues?, pinhole: PinholeCamera) -> InstanceRegion {
        let w = instance.maskWidth, h = instance.maskHeight
        let outline = MaskOutline.polygon(of: instance.mask, width: w, height: h, epsilon: 1.0)
        let centroid = MaskOutline.centroid(of: instance.mask, width: w, height: h)
        var points: [SIMD3<Float>] = []
        if let depth {
            let near = objects.tracker.depthMetres.first ?? 0.3, far = objects.tracker.depthMetres.last ?? 6
            // Sample the mask on a grid coarse enough to stay under the point budget.
            let step = max(1, Int((Double(instance.area) / Double(pointsPerInstance)).squareRoot().rounded(.up)))
            points.reserveCapacity(instance.area / (step * step) + 1)
            var y = step / 2
            while y < h {
                var x = step / 2
                while x < w {
                    // Only well inside the mask: its edge pixels are as likely to be what is behind.
                    if instance.inside(x: x, y: y), instance.inside(x: x - 1, y: y), instance.inside(x: x + 1, y: y),
                       instance.inside(x: x, y: y - 1), instance.inside(x: x, y: y + 1) {
                        let xu = (Float(x) + 0.5) / Float(w), yu = (Float(y) + 0.5) / Float(h)
                        // Upright to the sensor's landscape image.
                        let xs = yu, ys = 1 - xu
                        if let z = depth.metres(x: xs, y: ys), z >= near, z <= far {
                            points.append(pinhole.worldPoint(u: xs * Float(pinhole.width), v: ys * Float(pinhole.height), depth: z))
                        }
                    }
                    x += step
                }
                y += step
            }
        }
        // Where the thing is, for a vertex the depth map cannot answer: the middle of its own points.
        let distances = points.compactMap { pinhole.project($0)?.depth }.sorted()
        let typical = distances.isEmpty ? Float(2.5) : distances[distances.count / 2]
        let lifted = OutlineLift.lift(outline: outline, centroid: centroid, camera: pinhole, fallback: typical) { x, y in
            depth?.metres(x: x, y: y)
        }
        return InstanceRegion(classIndex: instance.classIndex, confidence: instance.confidence, outline: outline,
                              centroid: centroid, share: Double(instance.share), points: points,
                              worldOutline: lifted.outline, worldCentroid: lifted.centroid)
    }

    // MARK: Depth

    private struct DepthValues {
        var data: [Float]
        var width: Int
        var height: Int
        var fit: DepthScale.Fit?

        /// Nearest value at a normalised position of the sensor image.
        func at(x: Float, y: Float) -> Float? {
            let px = Int(x * Float(width)), py = Int(y * Float(height))
            guard px >= 0, py >= 0, px < width, py < height else { return nil }
            return data[py * width + px]
        }

        func scaled(_ fit: DepthScale.Fit) -> DepthValues {
            var copy = self
            copy.fit = fit
            return copy
        }

        /// Metres at a normalised position, once a fit is attached.
        func metres(x: Float, y: Float) -> Float? {
            guard let fit, let d = at(x: x, y: y) else { return nil }
            return fit.metres(d)
        }
    }

    /// The depth model's grayscale float16 image as floats.
    private static func floats(from buffer: CVPixelBuffer) -> DepthValues? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        var data = [Float](repeating: 0, count: width * height)
        switch CVPixelBufferGetPixelFormatType(buffer) {
        case kCVPixelFormatType_OneComponent16Half:
            for y in 0..<height {
                let row = (base + y * stride).assumingMemoryBound(to: Float16.self)
                for x in 0..<width { data[y * width + x] = Float(row[x]) }
            }
        case kCVPixelFormatType_OneComponent32Float:
            for y in 0..<height {
                let row = (base + y * stride).assumingMemoryBound(to: Float.self)
                for x in 0..<width { data[y * width + x] = row[x] }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<height {
                let row = (base + y * stride).assumingMemoryBound(to: UInt8.self)
                for x in 0..<width { data[y * width + x] = Float(row[x]) / 255 }
            }
        default:
            return nil
        }
        return DepthValues(data: data, width: width, height: height, fit: nil)
    }

    private static func pinhole(_ camera: ARCamera) -> PinholeCamera {
        let k = camera.intrinsics
        return PinholeCamera(fx: k.columns.0.x, fy: k.columns.1.y, cx: k.columns.2.x, cy: k.columns.2.y,
                             width: Int(camera.imageResolution.width), height: Int(camera.imageResolution.height),
                             transform: camera.transform)
    }

    /// The scale that turns the model's values into metres, from the tracked
    /// points that land in this frame.
    private static func fitScale(_ values: DepthValues, pinhole: PinholeCamera, points: [SIMD3<Float>]) -> DepthScale.Fit? {
        var predicted: [Float] = [], metres: [Float] = []
        for p in points {
            guard let (u, v, z) = pinhole.project(p), u >= 0, v >= 0, u < Float(pinhole.width), v < Float(pinhole.height),
                  let d = values.at(x: u / Float(pinhole.width), y: v / Float(pinhole.height)) else { continue }
            predicted.append(d)
            metres.append(z)
        }
        return DepthScale.fit(predicted: predicted, metres: metres)
    }
}
