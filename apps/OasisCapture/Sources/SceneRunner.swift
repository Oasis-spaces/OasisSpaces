import ARKit
import CoreImage
import CoreML
import Vision
import CaptureRules

/// What one analysed frame tells us.
struct FrameUnderstanding {
    var time: Double
    /// Upright (portrait) regions and class map.
    var segmentation: SegmentationResult
    /// How the depth model's output was scaled to metres, if it could be.
    var depthFit: DepthScale.Fit?
    /// World points on outlined things, with their class.
    var labelledPoints: [(SIMD3<Float>, Int)]
    /// Glowing edges tinted by class, in the sensor image's own orientation.
    var glow: CGImage?
}

/// Runs the room segmentation and the depth model on camera frames a few
/// times a second, one frame at a time, off the capture queue.
///
/// Depth is what makes objects hold still. The depth model gives a value for
/// every pixel (relative, not metres); the tracking's feature points give the
/// true depth at a few dozen pixels, so a scale is fitted per frame and every
/// pixel of an outlined object becomes a 3D point. From any angle those points
/// land on the same object in the room, so it fuses into one box instead of a
/// new one per viewpoint.
final class SceneRunner {
    let spec = DetectionSpec.bundled()
    /// Seconds between runs.
    var interval: Double = 0.3
    /// Pixels between depth samples on the depth map.
    var sampleStep = 6

    private let segmentation: VNCoreMLRequest?
    private let depth: VNCoreMLRequest?
    private let queue = DispatchQueue(label: "capture.scene", qos: .userInitiated)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var busy = false
    private var lastRun: Double = -1
    private var latest: SegmentationResult?
    private let lock = NSLock()

    init() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all   // the Neural Engine where there is one
        func request(_ name: String) -> VNCoreMLRequest? {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
                  let model = try? MLModel(contentsOf: url, configuration: configuration),
                  let visionModel = try? VNCoreMLModel(for: model) else { return nil }
            let request = VNCoreMLRequest(model: visionModel)
            // The whole frame, squeezed to the model's input: results map back by scaling.
            request.imageCropAndScaleOption = .scaleFill
            return request
        }
        segmentation = request("RoomSegmentation")
        depth = request("RoomDepth")
    }

    var isAvailable: Bool { segmentation != nil }
    var hasDepth: Bool { depth != nil }

    /// The latest segmentation, safe to read from any queue.
    var current: SegmentationResult? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    /// Analyses this frame if the previous run finished and the interval has
    /// passed. `done` gets the result on the scene queue.
    func submit(_ frame: ARFrame, glow wantGlow: Bool, done: @escaping (FrameUnderstanding) -> Void) {
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

            // 1. Classes, on the upright image.
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

            // 2. Depth, on the sensor image as it is (landscape, like the model was trained).
            var understanding = FrameUnderstanding(time: time, segmentation: result, depthFit: nil,
                                                   labelledPoints: [], glow: nil)
            if let depthRequest = self.depth,
               (try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([depthRequest])) != nil,
               let depthMap = (depthRequest.results?.first as? VNPixelBufferObservation)?.pixelBuffer,
               let values = Self.floats(from: depthMap) {
                let pinhole = Self.pinhole(camera)
                let fit = Self.fitScale(values, pinhole: pinhole, points: points)
                understanding.depthFit = fit
                if let fit, fit.error < 0.2 {
                    understanding.labelledPoints = self.backProject(values, fit: fit, pinhole: pinhole, classes: result)
                }
            }

            // 3. The glow.
            if wantGlow {
                understanding.glow = self.glowImage(buffer, classes: result)
            }
            done(understanding)
        }
    }

    // MARK: Depth

    private struct DepthValues {
        var data: [Float]
        var width: Int
        var height: Int

        /// Nearest value at a normalised position of the sensor image.
        func at(x: Float, y: Float) -> Float? {
            let px = Int(x * Float(width)), py = Int(y * Float(height))
            guard px >= 0, py >= 0, px < width, py < height else { return nil }
            return data[py * width + px]
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
        return DepthValues(data: data, width: width, height: height)
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

    /// A grid of depth samples on outlined things, as world points with their class.
    private func backProject(_ values: DepthValues, fit: DepthScale.Fit, pinhole: PinholeCamera,
                             classes: SegmentationResult) -> [(SIMD3<Float>, Int)] {
        let near = spec.pointDepthMetres.first ?? 0.3, far = spec.pointDepthMetres.last ?? 6
        var out: [(SIMD3<Float>, Int)] = []
        out.reserveCapacity(values.width * values.height / (sampleStep * sampleStep))
        let step = Double(2.5) / Double(classes.width)
        var py = sampleStep / 2
        while py < values.height {
            var px = sampleStep / 2
            while px < values.width {
                let xs = Float(px) / Float(values.width), ys = Float(py) / Float(values.height)
                // Sensor (landscape) coordinates to the upright class map.
                let xu = Double(1 - ys), yu = Double(xs)
                if let cls = classes.classAt(x: xu, y: yu), let info = spec.info(cls), info.outline,
                   info.group != "structure",
                   // Only well inside a region: the class map is coarse and an
                   // object's edge pixels are as likely to be the wall behind it.
                   [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)].allSatisfy({ classes.classAt(x: xu + $0, y: yu + $1) == cls }),
                   let z = fit.metres(values.data[py * values.width + px]), z >= near, z <= far {
                    let u = xs * Float(pinhole.width), v = ys * Float(pinhole.height)
                    out.append((pinhole.worldPoint(u: u, v: v, depth: z), cls))
                }
                px += sampleStep
            }
            py += sampleStep
        }
        return out
    }

    // MARK: Glow

    private static let glowLongSide: CGFloat = 512

    /// Edges of the camera image, lit in the colour of the class they belong
    /// to: what the eye reads as "the phone sees this".
    private func glowImage(_ buffer: CVPixelBuffer, classes: SegmentationResult) -> CGImage? {
        let source = CIImage(cvPixelBuffer: buffer)
        let scale = Self.glowLongSide / max(source.extent.width, source.extent.height)
        let small = source.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        // Edges, with texture and noise cut away (the contrast step pushes faint
        // edges to black), then thickened a touch so they read as lines.
        let edges = small.applyingFilter("CIEdges", parameters: [kCIInputIntensityKey: 3.0])
            .applyingFilter("CIColorControls", parameters: [kCIInputContrastKey: 3.0, kCIInputBrightnessKey: -0.35])
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 1.0])
        let soft = edges.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 2.0])
            .cropped(to: small.extent)
        let lit = edges.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: soft])
        // The edge brightness becomes the alpha of a class-coloured image.
        guard let colours = classColourImage(classes) else { return nil }
        let tint = CIImage(cgImage: colours).samplingNearest()
            .transformed(by: CGAffineTransform(scaleX: small.extent.width / CGFloat(colours.width),
                                               y: small.extent.height / CGFloat(colours.height)))
        let clear = CIImage(color: .clear).cropped(to: small.extent)
        let out = tint.applyingFilter("CIBlendWithMask", parameters: [kCIInputMaskImageKey: lit,
                                                                       kCIInputBackgroundImageKey: clear])
        return ciContext.createCGImage(out, from: small.extent)
    }

    /// The class map as a landscape RGBA image in the sensor's orientation,
    /// each pixel its class's colour (clear for classes not outlined).
    private func classColourImage(_ classes: SegmentationResult) -> CGImage? {
        let w = classes.height, h = classes.width   // upright map turned to landscape
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        var colourOf: [Int: (UInt8, UInt8, UInt8)] = [:]
        let structureIds = Set(spec.classes.filter { $0.group == "structure" }.map(\.id))
        for info in spec.classes where info.outline {
            let hex = spec.groups[info.group]?.color ?? "#FFFFFF"
            var value: UInt64 = 0
            Scanner(string: hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))).scanHexInt64(&value)
            colourOf[info.id] = (UInt8((value >> 16) & 0xFF), UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF))
        }
        for y in 0..<h {
            for x in 0..<w {
                // Sensor pixel (x, y) shows upright pixel (1 - y', x').
                let xu = classes.width - 1 - y * classes.width / h, yu = x * classes.height / w
                let cls = Int(classes.classes[yu * classes.width + xu])
                guard let (r, g, b) = colourOf[cls] else { continue }
                // Walls, floor and ceiling glow softly; things in the room glow fully.
                let a: UInt8 = structureIds.contains(cls) ? 90 : 255
                let i = (y * w + x) * 4
                bytes[i] = UInt8(UInt16(r) * UInt16(a) / 255); bytes[i + 1] = UInt8(UInt16(g) * UInt16(a) / 255)
                bytes[i + 2] = UInt8(UInt16(b) * UInt16(a) / 255); bytes[i + 3] = a
            }
        }
        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
