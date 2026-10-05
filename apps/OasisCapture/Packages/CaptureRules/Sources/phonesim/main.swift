// The phone's perception, run on the Mac over a processed space's frames, and
// judged against stage 3's room: the same Core ML models, the same decoding,
// refining, depth and tracking as Oasis Capture, fed the video's frames with
// the camera poses and sparse points the capture solve found (standing in for
// ARKit). Prints what was placed against the room's measured furniture.
//
//   cd apps/OasisCapture/Packages/CaptureRules
//   swift run -c release phonesim ../../../../spaces/<name> [--every 2] [--frames 80] [--debug] [--dump folder]
//
// Needs tools/phone_sim_export.py to have written spaces/<name>/phone-sim/frames.json.
import CoreImage
import CoreML
import ImageIO
import Foundation
import Vision
import CaptureRules

// MARK: Input

struct Frame: Decodable {
    var name: String
    var width: Int, height: Int
    var fx: Float, fy: Float, cx: Float, cy: Float
    var transform: [Float]          // column-major 4x4, camera to world (metres, y up)
    var points: [[Float]]
}

struct Truth: Decodable {
    struct Object: Decodable { var id: String; var label: String; var min: [Float]; var max: [Float] }
    struct Wall: Decodable { var center: [Float]; var along: [Float]; var normal: [Float]; var half: Float; var height: Float }
    struct Size: Decodable { var width: Float; var depth: Float; var height: Float }
    var space: String
    var images: String
    var room: Size
    var walls: [Wall]
    var objects: [Object]
    var frames: [Frame]
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    print("usage: phonesim <space folder> [--every n] [--frames n] [--debug] [--dump folder]")
    exit(2)
}
let space = URL(fileURLWithPath: arguments[1]).standardizedFileURL
func option(_ name: String, _ fallback: Int) -> Int {
    guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count, let v = Int(arguments[i + 1]) else { return fallback }
    return v
}
let every = option("--every", 1), limit = option("--frames", 1000)
let debug = arguments.contains("--debug")
/// Where to write each frame with its outlines, and the map (--dump <folder>).
let dumpFolder: URL? = arguments.firstIndex(of: "--dump").flatMap { i in
    guard i + 1 < arguments.count else { return nil }
    let url = URL(fileURLWithPath: arguments[i + 1])
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
let truth = try JSONDecoder().decode(Truth.self, from: Data(contentsOf: space.appendingPathComponent("phone-sim/frames.json")))
let resources = space.appendingPathComponent("../../apps/OasisCapture/Resources").standardizedFileURL
let spec = DetectionSpec.bundled(), objects = ObjectSpec.bundled()

// MARK: Models (compiled from the app's packages)

func load(_ name: String) throws -> MLModel {
    let compiled = try MLModel.compileModel(at: resources.appendingPathComponent("\(name).mlpackage"))
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    return try MLModel(contentsOf: compiled, configuration: configuration)
}
func request(_ model: MLModel) throws -> VNCoreMLRequest {
    let r = VNCoreMLRequest(model: try VNCoreMLModel(for: model))
    r.imageCropAndScaleOption = .scaleFill
    return r
}
let started = Date()
let detector = try request(try load("RoomObjects"))
let surfaces = try request(try load("RoomSegmentation"))
let encoder = try load("MaskEncoder"), decoder = try load("MaskDecoder")
let depthModel = try load("RoomMetricDepth")
print(String(format: "models loaded in %.1f s", Date().timeIntervalSince(started)))
let context = CIContext(options: [.cacheIntermediates: false])

/// A multi-array's values as a contiguous row-major float array in its shape's
/// order, whatever it holds and however it is laid out: Core ML pads some
/// outputs (a 3-channel point map stored 32 wide, a 518-pixel row stored 544
/// wide), so the strides must be honoured, not assumed.
func contiguousFloats(_ array: MLMultiArray) -> [Float] {
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

func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
    return buffer!
}

// MARK: The phone's steps, on one upright 3:4 frame

/// The frame as the phone's camera would hold it: cropped to 3:4 about the centre
/// (the video is 9:16; the sensor is 4:3), upright.
func upright(_ frame: Frame) -> (CIImage, PinholeCamera)? {
    let url = space.appendingPathComponent(truth.images).appendingPathComponent(frame.name)
    guard let image = CIImage(contentsOf: url) else { return nil }
    let w = image.extent.width, h = (image.extent.width * 4 / 3).rounded()
    let top = ((image.extent.height - h) / 2).rounded()
    // Core Image's origin is bottom-left: the crop's y runs from the bottom.
    let cropped = image.cropped(to: CGRect(x: 0, y: image.extent.height - top - h, width: w, height: h))
        .transformed(by: CGAffineTransform(translationX: 0, y: -(image.extent.height - top - h)))
    var transform = simd_float4x4()
    for c in 0..<4 { for r in 0..<4 { transform[c][r] = frame.transform[c * 4 + r] } }
    let camera = PinholeCamera(fx: frame.fx, fy: frame.fy, cx: frame.cx, cy: frame.cy - Float(top),
                               width: Int(w), height: Int(h), transform: transform)
    return (cropped, camera)
}

func decodeDetections(_ r: VNCoreMLRequest) -> [Instance] {
    guard let results = r.results as? [VNCoreMLFeatureValueObservation] else { return [] }
    var predictions: MLMultiArray?, protos: MLMultiArray?
    for o in results {
        guard let a = o.featureValue.multiArrayValue else { continue }
        if a.shape.count == 4 && a.shape[1].intValue == 32 { protos = a } else if a.shape.count == 3 { predictions = a }
    }
    guard let predictions, let protos else { return [] }
    return InstanceDecoder.decode(predictions: contiguousFloats(predictions), anchors: predictions.shape[2].intValue, protos: contiguousFloats(protos),
                                  maskWidth: protos.shape[3].intValue, maskHeight: protos.shape[2].intValue, spec: objects)
}

/// The surface model's class per pixel of the upright frame.
func surfaceClasses(_ r: VNCoreMLRequest) -> (classes: [Int32], width: Int, height: Int)? {
    guard let array = (r.results?.first as? VNCoreMLFeatureValueObservation)?.featureValue.multiArrayValue else { return nil }
    let shape = array.shape.map(\.intValue)
    let height = shape[shape.count - 2], width = shape[shape.count - 1]
    return (contiguousFloats(array).map { Int32($0) }, width, height)
}
let structure = spec.bareSurfaces
var onBareSurface = 0

let refinerSide: CGFloat = 1024
let refinerInput = pixelBuffer(width: 1024, height: 1024)
func refine(_ instances: [Instance], image: CIImage) -> [Instance] {
    let scale = refinerSide / max(image.extent.width, image.extent.height)
    let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    let square = CGRect(x: 0, y: 0, width: refinerSide, height: refinerSide)
    // Core Image puts y = 0 at the bottom: shift the image to the top of the square, like the phone.
    let lifted = scaled.transformed(by: CGAffineTransform(translationX: 0, y: refinerSide - scaled.extent.height))
    context.render(lifted.composited(over: CIImage(color: .black).cropped(to: square)), to: refinerInput, bounds: square, colorSpace: CGColorSpaceCreateDeviceRGB())
    guard let out = try? encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: refinerInput)])),
          let embedding = out.featureValue(for: "embedding") else { return instances }
    let imageWidth = Float(scaled.extent.width.rounded()), imageHeight = Float(scaled.extent.height.rounded())
    let maskWidth = Int((imageWidth / 4).rounded()), maskHeight = Int((imageHeight / 4).rounded())
    guard let box = try? MLMultiArray(shape: [1, 4], dataType: .float32) else { return instances }
    var refined = instances
    for i in refined.indices.prefix(6) where refined[i].share >= 0.004 {
        let inst = refined[i]
        box[0] = NSNumber(value: inst.minX * imageWidth); box[1] = NSNumber(value: inst.minY * imageHeight)
        box[2] = NSNumber(value: inst.maxX * imageWidth); box[3] = NSNumber(value: inst.maxY * imageHeight)
        guard let o = try? decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["embedding": embedding, "box": MLFeatureValue(multiArray: box)])),
              let logits = o.featureValue(for: "mask")?.multiArrayValue else { continue }
        let full = contiguousFloats(logits)
        let cols = logits.shape[logits.shape.count - 1].intValue
        let gx = (inst.maxX - inst.minX) * 0.1, gy = (inst.maxY - inst.minY) * 0.1
        let x0 = max(0, Int((inst.minX - gx) * Float(maskWidth))), x1 = min(maskWidth - 1, Int((inst.maxX + gx) * Float(maskWidth)))
        let y0 = max(0, Int((inst.minY - gy) * Float(maskHeight))), y1 = min(maskHeight - 1, Int((inst.maxY + gy) * Float(maskHeight)))
        guard x1 > x0, y1 > y0 else { continue }
        var mask = [UInt8](repeating: 0, count: maskWidth * maskHeight)
        for y in y0...y1 { for x in x0...x1 where full[y * cols + x] > 0 { mask[y * maskWidth + x] = 1 } }
        let area = MaskOutline.keepLargestComponent(&mask, width: maskWidth, height: maskHeight)
        guard Float(area) / Float(maskWidth * maskHeight) >= objects.minShare * 0.5 else { continue }
        refined[i].mask = mask; refined[i].maskWidth = maskWidth; refined[i].maskHeight = maskHeight; refined[i].area = area
        refined[i].quality = o.featureValue(for: "score")?.multiArrayValue.map { Float(truncating: $0[0]) }
    }
    return refined
}

/// Metres per pixel of the upright frame (nan where none), on a 392 x 518 grid
/// (the depth model takes the landscape sensor image; the frame is turned for it and back).
let depthWidth = 518, depthHeight = 392
let depthInput = pixelBuffer(width: depthWidth, height: depthHeight)
func metricDepth(_ image: CIImage, camera: PinholeCamera) -> (data: [Float], width: Int, height: Int)? {
    let landscape = image.oriented(.left)                 // upright -> the sensor's landscape (90 degrees anticlockwise)
    let scaled = landscape.transformed(by: CGAffineTransform(scaleX: CGFloat(depthWidth) / landscape.extent.width, y: CGFloat(depthHeight) / landscape.extent.height))
    context.render(scaled, to: depthInput, bounds: CGRect(x: 0, y: 0, width: depthWidth, height: depthHeight), colorSpace: CGColorSpaceCreateDeviceRGB())
    guard let out = try? depthModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: depthInput)])),
          let p = out.featureValue(for: "points")?.multiArrayValue, let m = out.featureValue(for: "mask")?.multiArrayValue,
          let s = out.featureValue(for: "metric_scale")?.multiArrayValue else { return nil }
    let points = contiguousFloats(p), mask = contiguousFloats(m), scale = Float(truncating: s[0])
    let focal = MetricDepth.relativeFocal(pixels: camera.fx, width: Float(camera.width), height: Float(camera.height))
    if debug {
        print("  points array: shape \(p.shape) strides \(p.strides) type \(p.dataType.rawValue); mask shape \(m.shape) strides \(m.strides); first 6 raw \(Array(points.prefix(6)))")
        print(String(format: "  depth input: upright extent %@, landscape extent %@, scaled extent %@; mask > 0.5 on %.0f%%, metric scale %.3f, focal rel %.3f, points z %.2f..%.2f",
                     "\(image.extent)", "\(landscape.extent)", "\(scaled.extent)", Double(mask.filter { $0 > 0.5 }.count) / Double(mask.count) * 100, scale, focal,
                     stride(from: 2, to: points.count, by: 3).map { points[$0] }.min() ?? 0, stride(from: 2, to: points.count, by: 3).map { points[$0] }.max() ?? 0))
        let dump = CIImage(cvPixelBuffer: depthInput)
        if let cg = context.createCGImage(dump, from: dump.extent) {
            let url = URL(fileURLWithPath: "/tmp/phonesim-depth-input.png")
            if let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) { CGImageDestinationAddImage(dest, cg, nil); CGImageDestinationFinalize(dest) }
        }
    }
    var depth = [Float](repeating: .nan, count: depthWidth * depthHeight)
    let ok: Bool = points.withUnsafeBufferPointer { pp in mask.withUnsafeBufferPointer { mm in
        guard let shift = MetricDepth.shift(points: pp.baseAddress!, mask: mm.baseAddress!, width: depthWidth, height: depthHeight, focal: focal) else { return false }
        if debug { print(String(format: "  shift %.3f (scale %.3f): depth = (z + shift) * scale", shift, scale)) }
        MetricDepth.metres(points: pp.baseAddress!, mask: mm.baseAddress!, width: depthWidth, height: depthHeight, shift: shift, scale: scale, into: &depth)
        return true
    } }
    guard ok else { return nil }
    // Back to upright: landscape (x, y) was upright (W-1-y... ) : upright pixel (xu, yu) sits at landscape (yu, H_l-1-xu)... derived below.
    var portrait = [Float](repeating: .nan, count: depthWidth * depthHeight)   // width depthHeight, height depthWidth
    let pw = depthHeight, ph = depthWidth
    for y in 0..<ph {
        for x in 0..<pw {
            // oriented(.left) turns the upright image anticlockwise: upright (x, y) -> landscape (y, W_u - 1 - x) in a W_u x H_u upright image.
            let lx = Int(Float(y) / Float(ph) * Float(depthWidth)), ly = Int(Float(pw - 1 - x) / Float(pw) * Float(depthHeight))
            portrait[y * pw + x] = depth[ly * depthWidth + lx]
        }
    }
    return (portrait, pw, ph)
}

/// The line that turns the model's inverse depth into the tracking points' (as on the phone:
/// points hidden behind the surface seen at their pixel are dropped and the fit made again).
func fitScale(_ depth: (data: [Float], width: Int, height: Int), camera: PinholeCamera, points: [SIMD3<Float>]) -> DepthScale.Fit? {
    var predicted: [Float] = [], metres: [Float] = []
    for p in points {
        guard let (u, v, z) = camera.project(p), z < 10, u >= 0, v >= 0, u < Float(camera.width), v < Float(camera.height) else { continue }
        let px = Int(u / Float(camera.width) * Float(depth.width)), py = Int(v / Float(camera.height) * Float(depth.height))
        let d = depth.data[py * depth.width + px]
        guard d > 0, d.isFinite else { continue }
        predicted.append(1 / d)
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

// MARK: Run

let builder = RoomMapBuilder(spec: objects)
for wall in truth.walls {
    builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(wall.center[0], wall.center[1], wall.center[2]),
                                    xAxis: SIMD3(wall.along[0], wall.along[1], wall.along[2]), zAxis: SIMD3(0, 1, 0), extent: SIMD2(2 * wall.half, wall.height)))
}
// The floor, as the phone's tracking would have found it.
builder.update(plane: PlaneInfo(id: UUID(), kind: .floor, vertical: false, center: SIMD3(0, 0, 0), xAxis: SIMD3(1, 0, 0),
                                zAxis: SIMD3(0, 0, 1), extent: SIMD2(truth.room.width, truth.room.depth)))
var recent: [SIMD3<Float>] = []
var framesRun = 0, framesWithDepth = 0, framesWithFit = 0, observationsTotal = 0, analysisTime = 0.0
var labelsSeen: [String: Int] = [:]
var records: [String] = []
let frames = Array(truth.frames.enumerated().filter { $0.offset % every == 0 }.map(\.element).prefix(limit))
for (n, frame) in frames.enumerated() {
    guard let (image, camera) = upright(frame) else { continue }
    let t = Date()
    let handler = VNImageRequestHandler(ciImage: image, orientation: .up)
    try? handler.perform([detector, surfaces])
    var instances = decodeDetections(detector)
    instances = refine(instances, image: image)
    // Furniture the surface model sees as bare wall is not shown or placed, as on the phone.
    let surfaceMap = surfaceClasses(surfaces)
    if let surfaceMap {
        let before = instances.count
        instances.removeAll { objects.isOnBareSurface($0, bare: structure, classes: surfaceMap.classes, width: surfaceMap.width, height: surfaceMap.height) }
        onBareSurface += before - instances.count
    }
    let onStructure = instances.map { inst in surfaceMap.map { inst.share(on: structure, classes: $0.classes, width: $0.width, height: $0.height) } ?? 0 }
    let fresh = frame.points.map { SIMD3<Float>($0[0], $0[1], $0[2]) }
    recent.append(contentsOf: fresh)
    if recent.count > 4000 { recent.removeFirst(recent.count - 4000) }
    var fit: DepthScale.Fit?
    var observations: [ObjectObservation] = []
    var lastRatio: Float = -1
    if let depth = metricDepth(image, camera: camera) {
        framesWithDepth += 1
        if debug {
            let finite = depth.data.filter { $0.isFinite && $0 > 0 }.sorted()
            print(String(format: "  depth: %d x %d, %.0f%% finite, min %.2f median %.2f max %.2f", depth.width, depth.height,
                         Double(finite.count) / Double(depth.data.count) * 100, finite.first ?? -1, finite.isEmpty ? -1 : finite[finite.count / 2], finite.last ?? -1))
            var ratios: [Float] = []
            var seen = 0
            for pt in fresh {
                guard let (u, v, z) = camera.project(pt), u >= 0, v >= 0, u < Float(camera.width), v < Float(camera.height) else { continue }
                seen += 1
                let d = depth.data[Int(v / Float(camera.height) * Float(depth.height)) * depth.width + Int(u / Float(camera.width) * Float(depth.width))]
                if d.isFinite, d > 0 { ratios.append(d / z) }
            }
            ratios.sort()
            print("  sparse points: \(fresh.count), in view \(seen), with depth \(ratios.count), predicted/true median \(ratios.isEmpty ? -1 : ratios[ratios.count / 2])")
        }
        var ratioMedian: Float = -1
        if dumpFolder != nil {
            var ratios: [Float] = []
            for pt in fresh {
                guard let (u, v, z) = camera.project(pt), u >= 0, v >= 0, u < Float(camera.width), v < Float(camera.height) else { continue }
                let d = depth.data[Int(v / Float(camera.height) * Float(depth.height)) * depth.width + Int(u / Float(camera.width) * Float(depth.width))]
                if d.isFinite, d > 0 { ratios.append(d / z) }
            }
            ratios.sort()
            if !ratios.isEmpty { ratioMedian = ratios[ratios.count / 2] }
        }
        lastRatio = ratioMedian
        let fitted = fitScale(depth, camera: camera, points: recent)
        if debug, let fitted { print(String(format: "  fit: a %.2f b %.2f samples %d error %.2f", fitted.a, fitted.b, fitted.samples, fitted.error)) } else if debug { print("  fit: none") }
        let (used, trusted) = DepthScale.correction(for: fitted)
        if trusted { framesWithFit += 1 }
        fit = used
        for inst in instances {
            let w = inst.maskWidth, h = inst.maskHeight
            var points: [SIMD3<Float>] = []
            // Things are placed only from depth the tracking points have corrected, as on the phone.
            for sample in (trusted ? inst.interiorSamples(budget: 500) : []) {
                let d = depth.data[Int(sample.y * Float(depth.height)) * depth.width + Int(sample.x * Float(depth.width))]
                if d > 0, d.isFinite, let z = used.metres(1 / d), z >= 0.3, z <= 6 {
                    points.append(camera.worldPoint(u: sample.x * Float(camera.width), v: sample.y * Float(camera.height), depth: z))
                }
            }
            observations.append(ObjectObservation(classIndex: inst.classIndex, confidence: inst.confidence, points: points))
            if debug, let label = objects.info(inst.classIndex)?.label {
                let xs = points.map(\.x), ys = points.map(\.y), zs = points.map(\.z)
                print(String(format: "  %@ (%.2f, mask %dx%d area %d): %d world points%@", label, inst.confidence, w, h, inst.area, points.count,
                             points.isEmpty ? "" : String(format: ", x %.2f..%.2f y %.2f..%.2f z %.2f..%.2f", xs.min()!, xs.max()!, ys.min()!, ys.max()!, zs.min()!, zs.max()!)))
            }
            if let label = objects.info(inst.classIndex)?.label { labelsSeen[label, default: 0] += 1 }
        }
    }
    let matches = builder.observe(observations, camera: camera)
    observationsTotal += observations.count
    if dumpFolder != nil {
        var items: [String] = []
        for (i, o) in observations.enumerated() {
            let inst = instances[i]
            let xs = o.points.map(\.x), ys = o.points.map(\.y), zs = o.points.map(\.z)
            let bounds = o.points.isEmpty ? "null" : String(format: "[[%.3f,%.3f,%.3f],[%.3f,%.3f,%.3f]]", xs.min()!, ys.min()!, zs.min()!, xs.max()!, ys.max()!, zs.max()!)
            let m = i < matches.count ? matches[i] : nil
            items.append(String(format: "{\"label\":\"%@\",\"conf\":%.2f,\"quality\":%@,\"share\":%.4f,\"structure\":%.2f,\"points\":%d,\"bounds\":%@,\"track\":%@,\"placed\":%@}",
                                objects.info(o.classIndex)?.label ?? "?", o.confidence, inst.quality.map { String(format: "%.2f", $0) } ?? "null", inst.share, onStructure[i], o.points.count, bounds,
                                m.map { "\"\($0.objectID)\"" } ?? "null", m?.placed == true ? "true" : "false"))
        }
        let fitText = fit.map { String(format: "{\"a\":%.3f,\"b\":%.3f,\"samples\":%d,\"error\":%.3f}", $0.a, $0.b, $0.samples, $0.error) } ?? "null"
        let cam = camera.transform.columns.3
        records.append(String(format: "{\"frame\":\"%@\",\"camera\":[%.3f,%.3f,%.3f],\"fit\":%@,\"modelOverTrue\":%.3f,\"instances\":[%@]}",
                              frame.name, cam.x, cam.y, cam.z, fitText, lastRatio, items.joined(separator: ",")))
    }
    if let dumpFolder {
        Draw.frame(image, context: context, instances: instances, matches: matches,
                   notes: onStructure.map { String(format: "s%.0f", $0 * 100) }, spec: objects,
                   to: dumpFolder.appendingPathComponent((frame.name as NSString).deletingPathExtension + ".jpg"))
    }
    analysisTime += Date().timeIntervalSince(t)
    framesRun += 1
    if n % 10 == 0 || n == frames.count - 1 {
        let placed = builder.build().objects
        print(String(format: "frame %3d/%d %@: %d things, depth %@, placed %d [%@]", n + 1, frames.count, frame.name, instances.count,
                     fit.map { $0.samples > 0 ? String(format: "fit %.2f from %d pts", $0.error, $0.samples) : "model" } ?? "none",
                     placed.count, placed.prefix(6).map(\.label).joined(separator: ", ")))
        _ = matches
    }
}

// MARK: Judge

let map = builder.build()
print(String(format: "\n%@: %d frames, %.0f ms a frame on this Mac, depth on %d, fitted on %d; %d observations",
             truth.space, framesRun, analysisTime / Double(max(1, framesRun)) * 1000, framesWithDepth, framesWithFit, observationsTotal))
print("\(onBareSurface) detections were on bare wall, floor or ceiling and dropped")
print("placed \(map.objects.count) objects; room yaw \(map.roomYaw.map { String(format: "%.2f", $0) } ?? "none")")
func overlap(_ aMin: SIMD3<Float>, _ aMax: SIMD3<Float>, _ bMin: SIMD3<Float>, _ bMax: SIMD3<Float>) -> Float {
    let w = max(0, min(aMax.x, bMax.x) - max(aMin.x, bMin.x)), d = max(0, min(aMax.z, bMax.z) - max(aMin.z, bMin.z))
    let inter = w * d
    let union = (aMax.x - aMin.x) * (aMax.z - aMin.z) + (bMax.x - bMin.x) * (bMax.z - bMin.z) - inter
    return union > 0 ? inter / union : 0
}
var matchedPlaced = Set<String>()
for object in truth.objects {
    let tMin = SIMD3(object.min[0], object.min[1], object.min[2]), tMax = SIMD3(object.max[0], object.max[1], object.max[2])
    var best: (ObjectBox, Float)?
    for placed in map.objects {
        let iou = overlap(placed.min, placed.max, tMin, tMax)
        if iou > (best?.1 ?? 0) { best = (placed, iou) }
    }
    if let (placed, iou) = best, iou > 0.05 {
        matchedPlaced.insert(placed.id)
        let centreError = simd_distance(SIMD2(placed.center.x, placed.center.z), SIMD2((tMin.x + tMax.x) / 2, (tMin.z + tMax.z) / 2))
        print(String(format: "  %@ %-16@ -> %-16@ footprint IoU %.2f, centre off %.2f m, size %.2fx%.2fx%.2f vs truth %.2fx%.2fx%.2f",
                     object.id, object.label, placed.label, iou, centreError, placed.size.x, placed.size.y, placed.size.z,
                     tMax.x - tMin.x, tMax.y - tMin.y, tMax.z - tMin.z))
    } else {
        print("  \(object.id) \(object.label): NOT PLACED")
    }
}
print("placed:")
for o in map.objects {
    print(String(format: "  %-5@ %-16@ x %+.2f..%+.2f  y %.2f..%.2f  z %+.2f..%+.2f  (%.2f x %.2f x %.2f)", o.id, o.label,
                 o.min.x, o.max.x, o.min.y, o.max.y, o.min.z, o.max.z, o.max.x - o.min.x, o.size.y, o.max.z - o.min.z))
}
let extra = map.objects.filter { !matchedPlaced.contains($0.id) }
print("  \(extra.count) placed objects match nothing measured: \(extra.map { "\($0.label) \(String(format: "%.1fx%.1f", $0.size.x, $0.size.z))" }.joined(separator: ", "))")
if let dumpFolder {
    Draw.map(map, truth: truth.objects, walls: truth.walls, to: dumpFolder.appendingPathComponent("map.png"))
    try? (records.joined(separator: "\n") + "\n").write(to: dumpFolder.appendingPathComponent("frames.jsonl"), atomically: true, encoding: .utf8)
}
print("labels seen: \(labelsSeen.sorted { $0.value > $1.value }.prefix(12).map { "\($0.key) \($0.value)" }.joined(separator: ", "))")
