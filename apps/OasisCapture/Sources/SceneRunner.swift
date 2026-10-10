import Accelerate
import ARKit
import Combine
import CoreImage
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
    /// The surface model saw bare wall here: not a thing of its own, unless it is a wardrobe's door.
    var doubtful = false
}

/// How far the models have loaded, for the preparing screen. Changed on the main queue.
final class ModelLoadState: ObservableObject {
    struct Step: Identifiable {
        let id: String
        let title: String
        var done = false
        var failed = false
    }

    @Published var steps = [
        Step(id: "RoomObjects", title: "Furniture and objects"),
        Step(id: "MaskEncoder", title: "Clean outlines"),
        Step(id: "RoomSegmentation", title: "Walls, floor and ceiling"),
        Step(id: "RoomMetricDepth", title: "Depth in metres"),
    ]
    @Published var ready = false

    var progress: Double { Double(steps.filter { $0.done || $0.failed }.count) / Double(steps.count) }
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
    /// Heights (world y) of points where the surface model saw floor, and of points all over the
    /// picture, from depth the tracking points corrected. The room map takes the floor from these.
    var floorAt: [Float] = []
    var pointsAt: [Float] = []
}

/// Runs the three models on camera frames a few times a second, one frame at
/// a time, off the capture queue:
///
/// - the object detector (YOLOE prompted with the room vocabulary) gives one
///   mask per thing: this sofa, that wardrobe, each pillow;
/// - the surface segmentation (SegFormer on ADE20K) gives the walls, floor,
///   ceiling, doors and windows, which a detector does not do well;
/// - the mask refiner (MobileSAM) turns each detected thing's box into one
///   clean whole-object mask, so the outline follows the real edges of a bed
///   or a wardrobe instead of the detector's blocky, fragmented guess;
/// - the depth model (MoGe-2 small) gives every pixel a place in metres on its
///   own: a point map plus a metric scale, with one depth offset recovered
///   from the camera's known focal length. Where ARKit's tracking points are
///   in view they correct it (a line fitted in inverse depth). Every pixel of a
///   detected thing becomes a 3D point; from any angle those points land on
///   the same object in the room, so it fuses into one box instead of a new
///   one per viewpoint.
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
    private var detector: VNCoreMLRequest?
    /// MoGe-2: a point map with its own metric scale (see MetricDepth).
    private var depth: MLModel?
    private var depthInput: CVPixelBuffer?
    /// The part of the camera's frame the models see, when it is not the whole frame (used on `queue`).
    private var analysedInput: CVPixelBuffer?
    private static let depthWidth = 518, depthHeight = 392
    private var maskEncoder: MLModel?
    private var maskDecoder: MLModel?
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var refinerInput: CVPixelBuffer?
    /// The refiner sees the upright image with its long side scaled to this, padded square.
    private static let refinerSide = 1024
    /// Most things refined per frame (the largest first), and the share of the image
    /// below which a thing keeps the detector's mask (a crisp edge on a plug socket is not
    /// worth a decode). On an iPhone 13 the encoder takes about 50 ms and each decode 12.
    var refineLimit = 6
    var refineMinShare: Float = 0.004
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
    /// Shown by the preparing screen while the models load.
    let loadState = ModelLoadState()
    /// Tracking points of recent frames (world space). One frame often has only a
    /// handful; the room's points of the last seconds are still where they were, and
    /// together they pin the depth scale far more often.
    private var recentPoints: [SIMD3<Float>] = []
    private static let recentPointLimit = 4000

    /// Loads the three models off the main thread, one after another (side by
    /// side they fight over the compiler: 114 s instead of 30 on a fresh
    /// install), each run once on a blank image so the first camera frame does
    /// not pay for the warm-up. The first load after an install compiles for
    /// the Neural Engine; later launches read the cache and take a second.
    func preload() {
        lock.lock()
        let start = !loading && segmentation == nil
        loading = true
        lock.unlock()
        guard start else { return }
        queue.async { [self] in
            let began = Date()
            var loaded: [String: VNCoreMLRequest] = [:]
            var models: [String: MLModel] = [:]
            func report(_ step: String, _ ok: Bool) {
                DispatchQueue.main.async {
                    if let i = self.loadState.steps.firstIndex(where: { $0.id == step }) {
                        self.loadState.steps[i].done = ok
                        self.loadState.steps[i].failed = !ok
                    }
                }
            }
            for name in ["RoomObjects", "MaskEncoder", "MaskDecoder", "RoomSegmentation", "RoomMetricDepth"] {
                let model = Self.model(name, units: [.all, .cpuAndGPU])
                if let model {
                    if name.hasPrefix("Mask") || name == "RoomMetricDepth" {
                        models[name] = model
                    } else if let request = Self.request(model) {
                        Self.warmUp(request, name: name)
                        loaded[name] = request
                    }
                }
                if name == "MaskDecoder" {
                    report("MaskEncoder", models["MaskEncoder"] != nil && models["MaskDecoder"] != nil)
                } else if name == "RoomMetricDepth" {
                    report(name, models[name] != nil)
                } else if !name.hasPrefix("Mask") {
                    report(name, loaded[name] != nil)
                }
            }
            lock.lock()
            segmentation = loaded["RoomSegmentation"]
            depth = models["RoomMetricDepth"]
            depthInput = Self.pixelBuffer(width: Self.depthWidth, height: Self.depthHeight)
            detector = loaded["RoomObjects"]
            maskEncoder = models["MaskEncoder"]
            maskDecoder = models["MaskDecoder"]
            refinerInput = Self.pixelBuffer(side: Self.refinerSide)
            loadSeconds = Date().timeIntervalSince(began)
            lock.unlock()
            AppLog.write(String(format: "models ready in %.1f s (surfaces %@, depth %@, objects %@, refiner %@)", loadSeconds,
                                segmentation == nil ? "missing" : "ok", depth == nil ? "missing" : "ok",
                                detector == nil ? "missing" : "ok",
                                maskEncoder != nil && maskDecoder != nil ? "ok" : "missing"))
            DispatchQueue.main.async { self.loadState.ready = true }
        }
    }

    /// One model, on the first of `units` that takes it.
    private static func model(_ name: String, units: [MLComputeUnits]) -> MLModel? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            AppLog.write("\(name) is not in the app bundle")
            return nil
        }
        for unit in units {
            let label = unit == .all ? "Neural Engine" : "GPU"
            let configuration = MLModelConfiguration()
            configuration.computeUnits = unit
            let started = Date()
            do {
                let model = try MLModel(contentsOf: url, configuration: configuration)
                AppLog.write(String(format: "%@ loaded in %.1f s (%@)", name, Date().timeIntervalSince(started), label))
                return model
            } catch {
                AppLog.write("\(name) failed to load (\(label)): \(error)")
            }
        }
        return nil
    }

    /// A Vision request for a model that takes the whole frame, squeezed to its input.
    private static func request(_ model: MLModel) -> VNCoreMLRequest? {
        guard let visionModel = try? VNCoreMLModel(for: model) else { return nil }
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
        return request
    }

    private static func pixelBuffer(side: Int) -> CVPixelBuffer? {
        pixelBuffer(width: side, height: side)
    }

    private static func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        return buffer
    }

    private static func name(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "not available"
        case .limited(.initializing): return "starting"
        case .limited(.excessiveMotion): return "limited (moving fast)"
        case .limited(.insufficientFeatures): return "limited (plain view)"
        case .limited(.relocalizing): return "limited (finding its place)"
        case .limited: return "limited"
        }
    }

    /// One run on a grey image: the first run of a model is many times slower than the rest.
    private static func warmUp(_ request: VNCoreMLRequest, name: String) {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, 640, 480, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard let buffer else { return }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, 128, CVPixelBufferGetDataSize(buffer))
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let started = Date()
        try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
        AppLog.write(String(format: "%@ warmed up in %.2f s", name, Date().timeIntervalSince(started)))
    }

    var isReady: Bool { lock.withLock { segmentation != nil || detector != nil } }
    var hasDepth: Bool { lock.withLock { depth != nil } }

    /// The latest surface segmentation, safe to read from any queue.
    var current: SegmentationResult? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    /// Analyses this frame if the previous run finished and the interval has
    /// passed. `done` gets the result on the scene queue.
    func submit(_ frame: ARFrame, done: @escaping (FrameUnderstanding) -> Void) {
        let (segmentation, depthModel, depthBuffer, detector) = lock.withLock { (self.segmentation, self.depth, self.depthInput, self.detector) }
        guard let segmentation, !busy, frame.timestamp - lastRun >= interval else { return }
        busy = true
        lastRun = frame.timestamp
        let buffer = frame.capturedImage
        let fresh = frame.rawFeaturePoints?.points ?? []
        recentPoints.append(contentsOf: fresh)
        if recentPoints.count > Self.recentPointLimit { recentPoints.removeFirst(recentPoints.count - Self.recentPointLimit) }
        let points = recentPoints
        let tracking = frame.camera.trackingState
        let camera = frame.camera
        let time = frame.timestamp
        queue.async { [weak self] in
            guard let self else { return }
            defer { self.busy = false }
            let began = Date()

            var marks: [(String, Date)] = [("start", began)]
            func mark(_ name: String) { marks.append((name, Date())) }
            // 0. The models take a 4:3 picture. At 4K the camera's is 16:9: its middle is analysed as it
            // is. (The whole of it squeezed in was a third out of shape for every model: on the test
            // videos the depth then placed furniture twice, 19 pieces where there are 8.)
            let (buffer, pinhole) = self.analysedView(of: buffer, camera: camera)
            mark("crop")
            // 1. Surfaces, on the upright image.
            let upright = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right)
            guard (try? upright.perform([segmentation])) != nil,
                  let observation = segmentation.results?.first as? VNCoreMLFeatureValueObservation,
                  let array = observation.featureValue.multiArrayValue else {
                if self.runs == 0 { AppLog.write("the surfaces model gave no result for a frame") }
                return
            }
            let shape = array.shape.map(\.intValue)
            let height = shape[shape.count - 2], width = shape[shape.count - 1]
            var classes = [Int32](repeating: 0, count: width * height)
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: width * height)
            classes.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: width * height) }
            let result = OutlineExtractor.extract(classes: classes, width: width, height: height, spec: self.spec)
            self.lock.lock()
            self.latest = result
            self.lock.unlock()
            mark("surfaces")

            // 2. Things, on the upright image.
            var instances: [Instance] = []
            if let detector {
                do {
                    try upright.perform([detector])
                    mark("detector")
                    instances = self.decode(detector)
                } catch {
                    if self.runs < 3 { AppLog.write("the object detector failed on a frame: \(error)") }
                }
            }
            mark("decode")
            var bare: [Bool] = []
            if !instances.isEmpty {
                instances = self.refine(instances, in: buffer)
                mark("refine")
            }

            // 3. Depth in metres, on the sensor image as it is (landscape).
            var understanding = FrameUnderstanding(time: time, segmentation: result, depthFit: nil,
                                                   instances: [], camera: pinhole)
            var depthValues: DepthValues?
            var trusted = false
            if let depthModel, let depthBuffer, let metric = self.metricDepth(depthModel, into: depthBuffer, from: buffer, pinhole: pinhole) {
                // Inverse depth, so the tracking points can correct it with the same line fit as before.
                var inverse = metric
                for i in inverse.indices { inverse[i] = metric.data[i] > 0 ? 1 / metric.data[i] : .nan }
                // With enough tracking points in view the fit corrects the model (measured: 6.6% error
                // instead of 11.7%) and things are placed in the room; without, the model's own metres
                // hang the outlines, and nothing is placed (see DepthScale.correction).
                let correction = DepthScale.correction(for: Self.fitScale(inverse, pinhole: pinhole, points: points))
                understanding.depthFit = correction.fit
                trusted = correction.trusted
                depthValues = inverse.scaled(correction.fit)
            }
            mark("depth")
            // Furniture the surface model sees as bare wall, and that stands out in depth from nothing,
            // is the detector seeing things: it is not shown or placed, unless the tracker finds it to
            // be a wardrobe's door. (Upright (x, y) is at sensor (y, 1 - x).)
            let surfaces = self.spec.bareSurfaces
            bare = instances.map { instance in
                self.objects.isOnBareSurface(instance, bare: surfaces, classes: classes, width: width, height: height) { x, y in
                    depthValues?.metres(x: y, y: 1 - x)
                }
            }
            if trusted, let depthValues {
                let near = self.objects.tracker.depthMetres.first ?? 0.3, far = self.objects.tracker.depthMetres.last ?? 6
                // Upright (x, y) is at sensor (y, 1 - x).
                func worldHeight(_ x: Float, _ y: Float) -> Float? {
                    let xs = y, ys = 1 - x
                    guard let z = depthValues.metres(x: xs, y: ys), z >= near, z <= far else { return nil }
                    return pinhole.worldPoint(u: xs * Float(pinhole.width), v: ys * Float(pinhole.height), depth: z).y
                }
                understanding.floorAt = self.spec.floorPositions(classes: classes, width: width, height: height).compactMap { worldHeight($0.x, $0.y) }
                for gy in 0..<24 {
                    for gx in 0..<24 {
                        if let h = worldHeight((Float(gx) + 0.5) / 24, (Float(gy) + 0.5) / 24) { understanding.pointsAt.append(h) }
                    }
                }
            }
            understanding.instances = instances.enumerated().map { i, instance in
                var region = self.region(instance, depth: depthValues, pinhole: pinhole, place: trusted)
                region.doubtful = i < bare.count && bare[i]
                return region
            }
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
            mark("lift")
            self.runs += 1
            if [1, 3, 10].contains(self.runs) || self.runs % 20 == 0 {
                let stages = zip(marks.dropFirst(), marks).map { String(format: "%@ %.0f", $0.0, $0.1.timeIntervalSince($1.1) * 1000) }
                let labels = instances.prefix(6).compactMap { inst -> String? in
                    guard let label = self.objects.info(inst.classIndex)?.label else { return nil }
                    return inst.quality.map { String(format: "%@ %.2f", label, $0) } ?? label
                }.joined(separator: ", ")
                AppLog.write(String(format: "analysis #%d: %.0f ms (%@) · %d things [%@] · depth %@ from %d points (%d in this frame, %d remembered) · tracking %@",
                                    self.runs, took * 1000, stages.joined(separator: ", "), instances.count, labels,
                                    understanding.depthFit.map { String(format: "fit %.2f", $0.error) } ?? "none",
                                    understanding.depthFit?.samples ?? 0, fresh.count, points.count, Self.name(tracking)))
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
        let p = Self.contiguousFloats(predictions), q = Self.contiguousFloats(protos)
        return InstanceDecoder.decode(predictions: p, anchors: anchors, protos: q, maskWidth: maskWidth,
                                      maskHeight: maskHeight, spec: objects)
    }

    // MARK: Depth in metres

    /// MoGe-2 on the landscape frame squeezed to its input: the point map's z
    /// plus the shift the known focal length implies, times the model's metric
    /// scale. Metres per pixel of the input, nan where the model has none.
    private func metricDepth(_ model: MLModel, into input: CVPixelBuffer, from buffer: CVPixelBuffer, pinhole: PinholeCamera) -> DepthValues? {
        let width = Self.depthWidth, height = Self.depthHeight
        let source = CIImage(cvPixelBuffer: buffer)
        let scaled = source.transformed(by: CGAffineTransform(scaleX: CGFloat(width) / source.extent.width,
                                                              y: CGFloat(height) / source.extent.height))
        ciContext.render(scaled, to: input, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: CGColorSpaceCreateDeviceRGB())
        let out: MLFeatureProvider
        do {
            out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
        } catch {
            if runs < 3 { AppLog.write("the depth model failed on a frame: \(error)") }
            return nil
        }
        guard let pointArray = out.featureValue(for: "points")?.multiArrayValue,
              let maskArray = out.featureValue(for: "mask")?.multiArrayValue,
              let scaleArray = out.featureValue(for: "metric_scale")?.multiArrayValue else { return nil }
        let points = Self.contiguousFloats(pointArray), mask = Self.contiguousFloats(maskArray)
        guard points.count == width * height * 3, mask.count == width * height else {
            if runs < 3 { AppLog.write("the depth model's outputs have an unexpected shape: \(pointArray.shape) \(maskArray.shape)") }
            return nil
        }
        let scale = Float(truncating: scaleArray[0])
        // The focal length relative to half the picture's diagonal (the input has the analysed picture's shape).
        let focal = MetricDepth.relativeFocal(pixels: pinhole.fx, width: Float(pinhole.width), height: Float(pinhole.height))
        var depth = [Float](repeating: .nan, count: width * height)
        let ok: Bool = points.withUnsafeBufferPointer { p in
            mask.withUnsafeBufferPointer { m in
                guard let shift = MetricDepth.shift(points: p.baseAddress!, mask: m.baseAddress!, width: width, height: height, focal: focal) else { return false }
                MetricDepth.metres(points: p.baseAddress!, mask: m.baseAddress!, width: width, height: height, shift: shift, scale: scale, into: &depth)
                return true
            }
        }
        guard ok else { return nil }
        return DepthValues(data: depth, width: width, height: height, fit: nil)
    }

    // MARK: The mask refiner

    /// Each instance's mask redone by the refiner: the upright image, scaled to
    /// the refiner's square, is encoded once; the detector's box of each thing
    /// (the largest first, up to `refineLimit`) is decoded into one clean mask
    /// at a quarter of the square, cut to the box grown by a tenth (the refiner
    /// may run on into a neighbour) and to its largest piece.
    private func refine(_ instances: [Instance], in buffer: CVPixelBuffer) -> [Instance] {
        let (encoder, decoder, input) = lock.withLock { (maskEncoder, maskDecoder, refinerInput) }
        guard let encoder, let decoder, let input else { return instances }
        let side = CGFloat(Self.refinerSide)
        let upright = CIImage(cvPixelBuffer: buffer).oriented(.right)
        let scale = side / max(upright.extent.width, upright.extent.height)
        let scaled = upright.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let square = CGRect(x: 0, y: 0, width: side, height: side)
        // The image in the square's corner; the rest black.
        let composed = scaled.composited(over: CIImage(color: .black).cropped(to: square))
        ciContext.render(composed, to: input, bounds: square, colorSpace: CGColorSpaceCreateDeviceRGB())
        let embedding: MLFeatureValue
        do {
            let out = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
            guard let value = out.featureValue(for: "embedding") else { return instances }
            embedding = value
        } catch {
            if runs < 3 { AppLog.write("the mask encoder failed on a frame: \(error)") }
            return instances
        }
        let imageWidth = Float(scaled.extent.width.rounded()), imageHeight = Float(scaled.extent.height.rounded())
        let maskWidth = Int((imageWidth / 4).rounded()), maskHeight = Int((imageHeight / 4).rounded())
        guard let box = try? MLMultiArray(shape: [1, 4], dataType: .float32) else { return instances }
        var refined = instances
        for i in refined.indices.prefix(refineLimit) where refined[i].share >= refineMinShare {
            let inst = refined[i]
            box[0] = NSNumber(value: inst.minX * imageWidth)
            box[1] = NSNumber(value: inst.minY * imageHeight)
            box[2] = NSNumber(value: inst.maxX * imageWidth)
            box[3] = NSNumber(value: inst.maxY * imageHeight)
            guard let out = try? decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "embedding": embedding, "box": MLFeatureValue(multiArray: box)])),
                  let logits = out.featureValue(for: "mask")?.multiArrayValue else { continue }
            let full = Self.contiguousFloats(logits)                       // 256 x 256 of the square
            let rows = logits.shape[logits.shape.count - 2].intValue, cols = logits.shape[logits.shape.count - 1].intValue
            guard rows >= maskHeight, cols >= maskWidth else { continue }
            // Cut to the box grown by a tenth of its size.
            let gx = (inst.maxX - inst.minX) * 0.1, gy = (inst.maxY - inst.minY) * 0.1
            let x0 = max(0, Int((inst.minX - gx) * Float(maskWidth))), x1 = min(maskWidth - 1, Int((inst.maxX + gx) * Float(maskWidth)))
            let y0 = max(0, Int((inst.minY - gy) * Float(maskHeight))), y1 = min(maskHeight - 1, Int((inst.maxY + gy) * Float(maskHeight)))
            guard x1 > x0, y1 > y0 else { continue }
            var mask = [UInt8](repeating: 0, count: maskWidth * maskHeight)
            var area = 0
            for y in y0...y1 {
                for x in x0...x1 where full[y * cols + x] > 0 {
                    mask[y * maskWidth + x] = 1
                    area += 1
                }
            }
            area = MaskOutline.keepLargestComponent(&mask, width: maskWidth, height: maskHeight)
            // A refined mask that lost nearly everything is the refiner missing the box: keep the detector's.
            guard Float(area) / Float(maskWidth * maskHeight) >= objects.minShare * 0.5 else { continue }
            refined[i].mask = mask
            refined[i].maskWidth = maskWidth
            refined[i].maskHeight = maskHeight
            refined[i].area = area
            refined[i].quality = out.featureValue(for: "score")?.multiArrayValue.map { Float(truncating: $0[0]) }
        }
        return refined
    }

    /// A multi-array's values as a contiguous row-major float array in its shape's
    /// order, whatever it holds and however it is laid out: Core ML pads some
    /// outputs (a 3-channel point map stored 32 wide, a 518-pixel row stored 544
    /// wide), so the strides must be honoured, not assumed.
    private static func contiguousFloats(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        let dims = shape.count
        // Fast path: already contiguous.
        var expected = 1
        var contiguous = true
        for d in stride(from: dims - 1, through: 0, by: -1) {
            if strides[d] != expected { contiguous = false; break }
            expected *= shape[d]
        }
        func read(_ body: (Int) -> Float) {
            if contiguous {
                for i in 0..<count { out[i] = body(i) }
                return
            }
            var index = [Int](repeating: 0, count: dims)
            for i in 0..<count {
                var offset = 0
                for d in 0..<dims { offset += index[d] * strides[d] }
                out[i] = body(offset)
                var d = dims - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
        }
        switch array.dataType {
        case .float32:
            let p = array.dataPointer.bindMemory(to: Float.self, capacity: count * 2)
            read { p[$0] }
        case .float16:
            let p = array.dataPointer.bindMemory(to: Float16.self, capacity: count * 2)
            read { Float(p[$0]) }
        case .double:
            let p = array.dataPointer.bindMemory(to: Double.self, capacity: count * 2)
            read { Float(p[$0]) }
        default:
            for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }

    /// An instance as an outline for the screen and, when the depth can be
    /// trusted to `place` things, world points for the room.
    private func region(_ instance: Instance, depth: DepthValues?, pinhole: PinholeCamera, place: Bool) -> InstanceRegion {
        let w = instance.maskWidth, h = instance.maskHeight
        let outline = MaskOutline.polygon(of: instance.mask, width: w, height: h, epsilon: 1.0)
        let centroid = MaskOutline.centroid(of: instance.mask, width: w, height: h)
        var points: [SIMD3<Float>] = []
        if let depth, place {
            let near = objects.tracker.depthMetres.first ?? 0.3, far = objects.tracker.depthMetres.last ?? 6
            for sample in instance.interiorSamples(budget: pointsPerInstance) {
                // Upright to the sensor's landscape image.
                let xs = sample.y, ys = 1 - sample.x
                if let z = depth.metres(x: xs, y: ys), z >= near, z <= far {
                    points.append(pinhole.worldPoint(u: xs * Float(pinhole.width), v: ys * Float(pinhole.height), depth: z))
                }
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

        var indices: Range<Int> { data.indices }
        subscript(i: Int) -> Float {
            get { data[i] }
            set { data[i] = newValue }
        }

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

    /// The shape of the picture the models take (the sensor's own, landscape).
    private static let analysedAspect: Float = 4.0 / 3

    /// The part of the camera's frame the models see, no more than 1440 across, and the camera
    /// that took just that part. The frame itself when it has the models' shape already.
    private func analysedView(of buffer: CVPixelBuffer, camera: ARCamera) -> (CVPixelBuffer, PinholeCamera) {
        let whole = Self.pinhole(camera)
        let (x, y, part) = whole.centred(aspect: Self.analysedAspect)
        guard part.width != whole.width || part.height != whole.height else { return (buffer, whole) }
        let scale = min(1, 1440 / CGFloat(part.width))
        let width = Int((CGFloat(part.width) * scale).rounded()), height = Int((CGFloat(part.height) * scale).rounded())
        if analysedInput == nil || CVPixelBufferGetWidth(analysedInput!) != width || CVPixelBufferGetHeight(analysedInput!) != height {
            analysedInput = Self.pixelBuffer(width: width, height: height)
        }
        guard let target = analysedInput else { return (buffer, whole) }
        // Core Image counts rows from the bottom.
        let rect = CGRect(x: x, y: whole.height - y - part.height, width: part.width, height: part.height)
        let image = CIImage(cvPixelBuffer: buffer).cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        ciContext.render(image, to: target, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: CGColorSpaceCreateDeviceRGB())
        return (target, part)
    }

    private static func pinhole(_ camera: ARCamera) -> PinholeCamera {
        let k = camera.intrinsics
        return PinholeCamera(fx: k.columns.0.x, fy: k.columns.1.y, cx: k.columns.2.x, cy: k.columns.2.y,
                             width: Int(camera.imageResolution.width), height: Int(camera.imageResolution.height),
                             transform: camera.transform)
    }

    /// The scale that turns the model's values into metres, from the tracked
    /// points that land in this frame. Remembered points can lie behind what
    /// this frame shows (the other side of a wardrobe): after a first fit,
    /// points much further than the surface seen at their pixel are dropped
    /// and the fit is made again.
    private static func fitScale(_ values: DepthValues, pinhole: PinholeCamera, points: [SIMD3<Float>]) -> DepthScale.Fit? {
        var predicted: [Float] = [], metres: [Float] = []
        for p in points {
            guard let (u, v, z) = pinhole.project(p), z < 10, u >= 0, v >= 0, u < Float(pinhole.width), v < Float(pinhole.height),
                  let d = values.at(x: u / Float(pinhole.width), y: v / Float(pinhole.height)) else { continue }
            predicted.append(d)
            metres.append(z)
        }
        guard let first = DepthScale.fit(predicted: predicted, metres: metres) else { return nil }
        var seenPredicted: [Float] = [], seenMetres: [Float] = []
        for (d, z) in zip(predicted, metres) {
            guard let surface = first.metres(d), z <= surface * 1.3, z >= surface * 0.6 else { continue }
            seenPredicted.append(d)
            seenMetres.append(z)
        }
        return DepthScale.fit(predicted: seenPredicted, metres: seenMetres) ?? first
    }
}
