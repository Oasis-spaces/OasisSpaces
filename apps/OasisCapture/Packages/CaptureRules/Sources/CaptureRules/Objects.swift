import Foundation
import simd
#if canImport(Accelerate)
import Accelerate
#endif

/// The object detector's classes and tuning (object-classes.json): the names
/// the model was prompted with, in the model's class order.
public struct ObjectSpec: Codable, Sendable {
    public struct ClassInfo: Codable, Sendable {
        public var id: Int
        public var prompt: String
        public var label: String
        public var group: String
        public var family: String
        /// Placed as a box in the room map (wall-mounted and small things are only outlined).
        public var boxed: Bool
        /// Always stands on the floor (a wardrobe, a bed; not a cabinet, which may hang on a wall).
        public var onFloor: Bool?
        /// Its top is at least, or at most, this high above the floor (a desk's top, a bathtub's rim).
        public var minTop: Float?
        public var maxTop: Float?
    }

    public struct Tracker: Codable, Sendable {
        public var voxelMetres: Float = 0.05
        public var confirmObservations: Int = 2
        public var matchOverlap: Float = 0.25
        public var mergeOverlap: Float = 0.4
        public var easing: Float = 0.35
        /// A bin on an axis counts towards the extent with this share of the busiest bin's hits.
        public var binShare: Float = 0.05
        /// The extent runs across empty gaps up to this long.
        public var gapMetres: Float = 0.15
        /// A part of a bed or wardrobe this close to the tracked one joins it.
        public var adjoinMetres: Float = 0.25
        /// Floor-standing furniture whose bottom is this close above the floor reaches it.
        public var snapMetres: Float = 0.5
        /// Furniture that always stands on the floor reaches it from this high (its bottom was hidden).
        public var floorReachMetres: Float = 1.0
        /// A standing piece's footprint is taken from what was seen of it below this height.
        public var bodyMetres: Float = 1.5
        /// Hits within this many analysed frames of each other are one glimpse.
        public var glimpseAnalyses: Int = 4
        /// Furniture whose back is this close to a wall reaches the wall.
        public var wallSnapMetres: Float = 0.2
        /// A wardrobe or appliance (seen only from the front) reaches a wall this far behind its front.
        public var unitDepthMetres: Float = 0.7
        /// Smaller than this on the floor (and under 0.4 m tall): not furniture, not placed.
        public var minBoxMetres: Float = 0.25
        public var minPoints: Int = 25
        public var staleObservations: Int = 6
        public var forgetObservations: Int = 24
        public var maxSizeMetres: Float = 4
        public var depthMetres: [Float] = [0.3, 6]

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            voxelMetres = try c.decodeIfPresent(Float.self, forKey: .voxelMetres) ?? voxelMetres
            confirmObservations = try c.decodeIfPresent(Int.self, forKey: .confirmObservations) ?? confirmObservations
            matchOverlap = try c.decodeIfPresent(Float.self, forKey: .matchOverlap) ?? matchOverlap
            mergeOverlap = try c.decodeIfPresent(Float.self, forKey: .mergeOverlap) ?? mergeOverlap
            easing = try c.decodeIfPresent(Float.self, forKey: .easing) ?? easing
            binShare = try c.decodeIfPresent(Float.self, forKey: .binShare) ?? binShare
            gapMetres = try c.decodeIfPresent(Float.self, forKey: .gapMetres) ?? gapMetres
            adjoinMetres = try c.decodeIfPresent(Float.self, forKey: .adjoinMetres) ?? adjoinMetres
            snapMetres = try c.decodeIfPresent(Float.self, forKey: .snapMetres) ?? snapMetres
            floorReachMetres = try c.decodeIfPresent(Float.self, forKey: .floorReachMetres) ?? floorReachMetres
            bodyMetres = try c.decodeIfPresent(Float.self, forKey: .bodyMetres) ?? bodyMetres
            glimpseAnalyses = try c.decodeIfPresent(Int.self, forKey: .glimpseAnalyses) ?? glimpseAnalyses
            wallSnapMetres = try c.decodeIfPresent(Float.self, forKey: .wallSnapMetres) ?? wallSnapMetres
            unitDepthMetres = try c.decodeIfPresent(Float.self, forKey: .unitDepthMetres) ?? unitDepthMetres
            minBoxMetres = try c.decodeIfPresent(Float.self, forKey: .minBoxMetres) ?? minBoxMetres
            minPoints = try c.decodeIfPresent(Int.self, forKey: .minPoints) ?? minPoints
            staleObservations = try c.decodeIfPresent(Int.self, forKey: .staleObservations) ?? staleObservations
            forgetObservations = try c.decodeIfPresent(Int.self, forKey: .forgetObservations) ?? forgetObservations
            maxSizeMetres = try c.decodeIfPresent(Float.self, forKey: .maxSizeMetres) ?? maxSizeMetres
            depthMetres = try c.decodeIfPresent([Float].self, forKey: .depthMetres) ?? depthMetres
        }
    }

    /// What it takes for a detection to be outlined on the screen (see SightingMemory).
    public struct Screen: Codable, Sendable {
        /// A thing is outlined from this many sightings on.
        public var showSightings: Int = 2
        /// A thing not seen for this many analysed frames is forgotten.
        public var forgetAnalyses: Int = 6
        /// Two sightings this close in direction are the same thing, however small it is.
        public var minAngleDegrees: Float = 6

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            showSightings = try c.decodeIfPresent(Int.self, forKey: .showSightings) ?? showSightings
            forgetAnalyses = try c.decodeIfPresent(Int.self, forKey: .forgetAnalyses) ?? forgetAnalyses
            minAngleDegrees = try c.decodeIfPresent(Float.self, forKey: .minAngleDegrees) ?? minAngleDegrees
        }
    }

    public var model: String
    /// Width and height of the model's input image.
    public var inputSize: [Int]
    public var confidence: Float
    public var iou: Float
    public var maskThreshold: Float
    /// A detection whose mask covers less of the image than this is dropped.
    public var minShare: Float
    /// A piece of furniture, a window or a curtain whose mask lies this much on what the
    /// surface model calls bare wall, floor or ceiling is the detector seeing things.
    public var bareSurfaceShare: Float
    public var kin: [String: [String]]
    public var tracker: Tracker
    public var screen: Screen
    public var classes: [ClassInfo]

    private enum CodingKeys: String, CodingKey {
        case model, inputSize, confidence, iou, maskThreshold, minShare, bareSurfaceShare, kin, tracker, screen, classes
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(String.self, forKey: .model)
        inputSize = try c.decode([Int].self, forKey: .inputSize)
        confidence = try c.decode(Float.self, forKey: .confidence)
        iou = try c.decode(Float.self, forKey: .iou)
        maskThreshold = try c.decode(Float.self, forKey: .maskThreshold)
        minShare = try c.decodeIfPresent(Float.self, forKey: .minShare) ?? 0.002
        bareSurfaceShare = try c.decodeIfPresent(Float.self, forKey: .bareSurfaceShare) ?? 0.9
        tracker = try c.decodeIfPresent(Tracker.self, forKey: .tracker) ?? Tracker()
        screen = try c.decodeIfPresent(Screen.self, forKey: .screen) ?? Screen()
        classes = try c.decode([ClassInfo].self, forKey: .classes)
        // The kin dictionary carries a comment string beside the lists.
        kin = [:]
        if let raw = try? c.decode([String: KinValue].self, forKey: .kin) {
            kin = raw.compactMapValues { if case .families(let f) = $0 { return f } else { return nil } }
        }
    }

    private enum KinValue: Decodable {
        case families([String])
        case comment(String)
        init(from decoder: Decoder) throws {
            let single = try decoder.singleValueContainer()
            if let list = try? single.decode([String].self) { self = .families(list) } else { self = .comment(try single.decode(String.self)) }
        }
    }

    public static func bundled() -> ObjectSpec {
        guard let url = Bundle.module.url(forResource: "object-classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let spec = try? JSONDecoder().decode(ObjectSpec.self, from: data) else {
            fatalError("object-classes.json is missing or does not match ObjectSpec")
        }
        return spec
    }

    public var inputWidth: Int { inputSize[0] }
    public var inputHeight: Int { inputSize[1] }

    public func info(_ index: Int) -> ClassInfo? {
        index >= 0 && index < classes.count ? classes[index] : nil
    }

    public func index(of name: String) -> Int? {
        classes.firstIndex { $0.prompt == name || $0.label == name }
    }

    /// Whether a detection is the detector seeing things on a bare surface: a
    /// fridge that is a stretch of white wall, a window on a plain wall, a
    /// curtain on a cupboard door. The surface model (`classes`, its class per
    /// pixel of the same upright image; `bare`, its wall, floor and ceiling
    /// classes) is asked what is under the mask. Only furniture, windows and
    /// curtains are checked: screens, pictures, lights and rugs lie flat on a
    /// surface by nature, and a door is often wall to the surface model.
    public func isOnBareSurface(_ instance: Instance, bare: Set<Int32>, classes: [Int32], width: Int, height: Int) -> Bool {
        guard let info = info(instance.classIndex) else { return false }
        let checked = (info.boxed && !["rug", "screen"].contains(info.family)) || ["window", "curtain"].contains(info.family)
        return checked && instance.share(on: bare, classes: classes, width: width, height: height) >= bareSurfaceShare
    }

    /// The kin group a family belongs to (the family itself when it has none):
    /// what counts as "the same object" for matching and labels.
    public func kinGroup(of family: String) -> String {
        kin.first { $0.value.contains(family) }?.key ?? family
    }

    public func kinGroup(_ index: Int) -> String? {
        info(index).map { kinGroup(of: $0.family) }
    }
}

/// One detected thing in a frame.
public struct Instance: Sendable {
    public var classIndex: Int
    public var confidence: Float
    /// Box in normalised model-input coordinates, x right, y down, 0...1.
    public var minX: Float, minY: Float, maxX: Float, maxY: Float
    /// Mask at the prototype resolution (row-major, maskWidth x maskHeight), 1 inside.
    public var mask: [UInt8]
    public var maskWidth: Int
    public var maskHeight: Int
    /// Mask pixels inside.
    public var area: Int
    /// The mask refiner's own estimate of its mask (0...1), when it refined this one.
    public var quality: Float? = nil

    public init(classIndex: Int, confidence: Float, minX: Float, minY: Float, maxX: Float, maxY: Float,
                mask: [UInt8], maskWidth: Int, maskHeight: Int, area: Int, quality: Float? = nil) {
        self.classIndex = classIndex; self.confidence = confidence
        self.minX = minX; self.minY = minY; self.maxX = maxX; self.maxY = maxY
        self.mask = mask; self.maskWidth = maskWidth; self.maskHeight = maskHeight; self.area = area
        self.quality = quality
    }

    /// Share of the image the mask covers.
    public var share: Float { Float(area) / Float(maskWidth * maskHeight) }

    public func inside(x: Int, y: Int) -> Bool {
        x >= 0 && y >= 0 && x < maskWidth && y < maskHeight && mask[y * maskWidth + x] != 0
    }
}

extension Instance {
    /// Places to read the depth of this thing: a grid of about `budget`
    /// samples, as normalised upright image coordinates, none within `margin`
    /// mask pixels of the mask's edge. The depth map is soft at an edge (a
    /// pixel there is somewhere between the thing and the wall behind it),
    /// and those in-between points stretch a box towards the wall.
    public func interiorSamples(budget: Int, margin: Int = 2) -> [SIMD2<Float>] {
        let step = max(1, Int((Double(area) / Double(max(1, budget))).squareRoot().rounded(.up)))
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(area / (step * step) + 1)
        var y = step / 2
        while y < maskHeight {
            var x = step / 2
            while x < maskWidth {
                if inside(x: x, y: y), inside(x: x - 1, y: y), inside(x: x + 1, y: y), inside(x: x, y: y - 1), inside(x: x, y: y + 1),
                   inside(x: x - margin, y: y), inside(x: x + margin, y: y), inside(x: x, y: y - margin), inside(x: x, y: y + margin) {
                    out.append(SIMD2((Float(x) + 0.5) / Float(maskWidth), (Float(y) + 0.5) / Float(maskHeight)))
                }
                x += step
            }
            y += step
        }
        return out
    }
}

extension Instance {
    /// Share of this thing's mask that lies on pixels the surface model calls
    /// one of `surfaces` (bare wall, floor, ceiling). `classes` is the surface
    /// model's class per pixel of the same upright image, row-major.
    public func share(on surfaces: Set<Int32>, classes: [Int32], width: Int, height: Int) -> Float {
        guard area > 0, width > 0, height > 0, classes.count >= width * height else { return 0 }
        var on = 0, all = 0
        for y in 0..<maskHeight {
            let cy = min(height - 1, Int((Float(y) + 0.5) / Float(maskHeight) * Float(height)))
            for x in 0..<maskWidth where mask[y * maskWidth + x] != 0 {
                let cx = min(width - 1, Int((Float(x) + 0.5) / Float(maskWidth) * Float(width)))
                all += 1
                if surfaces.contains(classes[cy * width + cx]) { on += 1 }
            }
        }
        return all > 0 ? Float(on) / Float(all) : 0
    }
}

/// Turns the detector's raw outputs into instances: the best class per anchor,
/// non-maximum suppression within a kin group (a sofa and the armchair the
/// model also sees on it are one thing), then a mask per kept detection from
/// the prototype masks, cut to its box and to its largest connected piece.
public enum InstanceDecoder {
    /// `predictions`: (4 + classes + 32) x anchors, channel-major, boxes as
    /// centre x, centre y, width, height in input pixels. `protos`: 32 x
    /// maskHeight x maskWidth. Returns instances largest first.
    public static func decode(predictions: UnsafePointer<Float>, anchors: Int,
                              protos: UnsafePointer<Float>, maskWidth: Int, maskHeight: Int,
                              spec: ObjectSpec, maxInstances: Int = 24) -> [Instance] {
        let nc = spec.classes.count
        let channels = 4 + nc + 32
        let width = Float(spec.inputWidth), height = Float(spec.inputHeight)
        func at(_ channel: Int, _ anchor: Int) -> Float { predictions[channel * anchors + anchor] }

        // 1. Candidates: the best class of every anchor above the threshold.
        struct Candidate { var anchor: Int; var cls: Int; var score: Float; var box: SIMD4<Float> }
        var candidates: [Candidate] = []
        for a in 0..<anchors {
            var best = -1, bestScore = spec.confidence
            for c in 0..<nc {
                let s = at(4 + c, a)
                if s >= bestScore { bestScore = s; best = c }
            }
            guard best >= 0 else { continue }
            let cx = at(0, a) / width, cy = at(1, a) / height, w = at(2, a) / width, h = at(3, a) / height
            guard w > 0, h > 0 else { continue }
            candidates.append(Candidate(anchor: a, cls: best, score: bestScore,
                                        box: SIMD4(max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2))))
        }
        candidates.sort { $0.score > $1.score }

        // 2. Suppression: within a kin group at the spec's IoU; across everything when nearly identical.
        var kept: [Candidate] = []
        for c in candidates where kept.count < maxInstances {
            let kin = spec.kinGroup(c.cls)
            var suppressed = false
            for k in kept {
                let overlap = iou(c.box, k.box)
                if overlap >= 0.85 || (overlap >= spec.iou && spec.kinGroup(k.cls) == kin) { suppressed = true; break }
            }
            if !suppressed { kept.append(c) }
        }
        guard !kept.isEmpty, channels * anchors > 0 else { return [] }

        // 3. Masks: sigmoid(coefficients . prototypes), inside the box, above the threshold.
        let pixels = maskWidth * maskHeight
        var coefficients = [Float](repeating: 0, count: kept.count * 32)
        for (i, c) in kept.enumerated() {
            for k in 0..<32 { coefficients[i * 32 + k] = at(4 + nc + k, c.anchor) }
        }
        var product = [Float](repeating: 0, count: kept.count * pixels)
        multiply(coefficients, rows: kept.count, protos, pixels: pixels, into: &product)

        let logit = log(spec.maskThreshold / (1 - spec.maskThreshold))   // sigmoid(x) > t  <=>  x > logit(t)
        var instances: [Instance] = []
        for (i, c) in kept.enumerated() {
            let x0 = Int(c.box.x * Float(maskWidth)), x1 = min(maskWidth - 1, Int(c.box.z * Float(maskWidth)))
            let y0 = Int(c.box.y * Float(maskHeight)), y1 = min(maskHeight - 1, Int(c.box.w * Float(maskHeight)))
            guard x1 >= x0, y1 >= y0 else { continue }
            var mask = [UInt8](repeating: 0, count: pixels)
            let base = i * pixels
            var area = 0
            for y in y0...y1 {
                for x in x0...x1 where product[base + y * maskWidth + x] > logit {
                    mask[y * maskWidth + x] = 1
                    area += 1
                }
            }
            // Keep the largest connected piece: a stray blob elsewhere in the box is not this object.
            area = MaskOutline.keepLargestComponent(&mask, width: maskWidth, height: maskHeight)
            guard Float(area) / Float(pixels) >= spec.minShare else { continue }
            instances.append(Instance(classIndex: c.cls, confidence: c.score,
                                      minX: c.box.x, minY: c.box.y, maxX: c.box.z, maxY: c.box.w,
                                      mask: mask, maskWidth: maskWidth, maskHeight: maskHeight, area: area))
        }
        return instances.sorted { $0.area > $1.area }
    }

    /// Convenience for arrays.
    public static func decode(predictions: [Float], anchors: Int, protos: [Float], maskWidth: Int, maskHeight: Int,
                              spec: ObjectSpec, maxInstances: Int = 24) -> [Instance] {
        predictions.withUnsafeBufferPointer { p in
            protos.withUnsafeBufferPointer { q in
                decode(predictions: p.baseAddress!, anchors: anchors, protos: q.baseAddress!,
                       maskWidth: maskWidth, maskHeight: maskHeight, spec: spec, maxInstances: maxInstances)
            }
        }
    }

    static func iou(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Float {
        let w = max(0, min(a.z, b.z) - max(a.x, b.x)), h = max(0, min(a.w, b.w) - max(a.y, b.y))
        let inter = w * h
        let union = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - inter
        return union > 0 ? inter / union : 0
    }

    /// product (rows x pixels) = coefficients (rows x 32) . protos (32 x pixels)
    private static func multiply(_ coefficients: [Float], rows: Int, _ protos: UnsafePointer<Float>, pixels: Int,
                                 into product: inout [Float]) {
        #if canImport(Accelerate)
        coefficients.withUnsafeBufferPointer { a in
            product.withUnsafeMutableBufferPointer { c in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(pixels), 32,
                            1, a.baseAddress!, 32, protos, Int32(pixels), 0, c.baseAddress!, Int32(pixels))
            }
        }
        #else
        for r in 0..<rows {
            for k in 0..<32 {
                let w = coefficients[r * 32 + k]
                guard w != 0 else { continue }
                for p in 0..<pixels { product[r * pixels + p] += w * protos[k * pixels + p] }
            }
        }
        #endif
    }
}

/// Outlines and clean-up of binary masks.
public enum MaskOutline {
    /// Keeps only the largest 4-connected component of the mask; returns its area.
    @discardableResult
    public static func keepLargestComponent(_ mask: inout [UInt8], width: Int, height: Int) -> Int {
        let total = width * height
        var label = [Int32](repeating: 0, count: total)
        var sizes: [Int] = [0]
        var queue: [Int] = []
        for start in 0..<total where mask[start] != 0 && label[start] == 0 {
            let id = Int32(sizes.count)
            sizes.append(0)
            queue.removeAll(keepingCapacity: true)
            queue.append(start)
            label[start] = id
            var head = 0
            while head < queue.count {
                let i = queue[head]
                head += 1
                sizes[Int(id)] += 1
                let x = i % width, y = i / width
                if x > 0, mask[i - 1] != 0, label[i - 1] == 0 { label[i - 1] = id; queue.append(i - 1) }
                if x < width - 1, mask[i + 1] != 0, label[i + 1] == 0 { label[i + 1] = id; queue.append(i + 1) }
                if y > 0, mask[i - width] != 0, label[i - width] == 0 { label[i - width] = id; queue.append(i - width) }
                if y < height - 1, mask[i + width] != 0, label[i + width] == 0 { label[i + width] = id; queue.append(i + width) }
            }
        }
        guard sizes.count > 1 else { return 0 }
        let largest = Int32(sizes.indices.dropFirst().max { sizes[$0] < sizes[$1] }!)
        for i in 0..<total where label[i] != largest { mask[i] = 0 }
        return sizes[Int(largest)]
    }

    /// The outer boundary of the mask's (single) piece as a simplified polygon
    /// in normalised coordinates (0...1, pixel centres). Empty for an empty mask.
    public static func polygon(of mask: [UInt8], width: Int, height: Int, epsilon: Double = 1.0) -> [SIMD2<Double>] {
        guard let start = mask.firstIndex(where: { $0 != 0 }) else { return [] }
        let boundary = OutlineExtractor.traceBoundary(start: start, width: width, height: height) { mask[$0] != 0 }
        let simplified = OutlineExtractor.simplifyClosed(boundary.map { SIMD2(Double($0 % width), Double($0 / width)) }, epsilon: epsilon)
        return simplified.map { SIMD2(($0.x + 0.5) / Double(width), ($0.y + 0.5) / Double(height)) }
    }

    /// Centre of the mask in normalised coordinates.
    public static func centroid(of mask: [UInt8], width: Int, height: Int) -> SIMD2<Double> {
        var sx = 0, sy = 0, n = 0
        for i in 0..<(width * height) where mask[i] != 0 { sx += i % width; sy += i / width; n += 1 }
        guard n > 0 else { return SIMD2(0.5, 0.5) }
        return SIMD2((Double(sx) / Double(n) + 0.5) / Double(width), (Double(sy) / Double(n) + 0.5) / Double(height))
    }
}
