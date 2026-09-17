import SwiftUI
import simd
import CaptureRules

/// The floor plan as the scan has it so far: floor, walls, placed furniture,
/// the path walked and where the phone is. Drawn small beside the camera and
/// full-size in the map sheet.
struct RoomMapCanvas: View {
    @ObservedObject var state: CaptureState
    let spec: DetectionSpec
    var detailed = false

    var body: some View {
        Canvas { context, size in
            let map = state.map
            // Fit everything known, with the phone, into the view (at least 3 m across).
            var lo = SIMD2(state.position.x - 1.5, state.position.y - 1.5)
            var hi = SIMD2(state.position.x + 1.5, state.position.y + 1.5)
            if let bounds = map.bounds { lo = simd_min(lo, bounds.min); hi = simd_max(hi, bounds.max) }
            for p in state.path { lo = simd_min(lo, p); hi = simd_max(hi, p) }
            let inset: CGFloat = detailed ? 24 : 14
            let scale = min((size.width - 2 * inset) / CGFloat(hi.x - lo.x), (size.height - 2 * inset) / CGFloat(hi.y - lo.y))
            let origin = CGPoint(x: (size.width - CGFloat(hi.x - lo.x) * scale) / 2, y: (size.height - CGFloat(hi.y - lo.y) * scale) / 2)
            func toMap(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: origin.x + CGFloat(p.x - lo.x) * scale, y: origin.y + CGFloat(p.y - lo.y) * scale)
            }
            func rect(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> CGRect {
                let p = toMap(a), q = toMap(b)
                return CGRect(x: min(p.x, q.x), y: min(p.y, q.y), width: abs(q.x - p.x), height: abs(q.y - p.y))
            }

            // Floor first, then scanned points faintly, then furniture, then walls on top.
            for floor in map.floors {
                var path = Path()
                let corners = floor.corners.map { toMap(SIMD2($0.x, $0.z)) }
                path.move(to: corners[0])
                for c in corners.dropFirst() { path.addLine(to: c) }
                path.closeSubpath()
                context.fill(path, with: .color(.white.opacity(0.10)))
                context.stroke(path, with: .color(.white.opacity(0.25)), lineWidth: 1)
            }
            if !detailed {
                for p in state.mapPoints {
                    let q = toMap(p)
                    context.fill(Path(ellipseIn: CGRect(x: q.x - 0.7, y: q.y - 0.7, width: 1.4, height: 1.4)),
                                 with: .color(.white.opacity(0.35)))
                }
            }
            for object in map.objects {
                let color = Color(hex: spec.groups[object.group]?.color ?? "#FFFFFF")
                let r = rect(object.footprintMin, object.footprintMax)
                let shape = Path(roundedRect: r, cornerRadius: min(4, r.width / 4))
                context.fill(shape, with: .color(color.opacity(0.55)))
                context.stroke(shape, with: .color(color), lineWidth: 1.5)
                if detailed || r.width > 28 {
                    let text = context.resolve(Text(object.label).font(.system(size: detailed ? 12 : 9, weight: .semibold)).foregroundColor(.white))
                    context.draw(text, at: CGPoint(x: r.midX, y: r.midY))
                }
            }
            for wall in map.walls {
                var line = Path()
                line.move(to: toMap(wall.from))
                line.addLine(to: toMap(wall.to))
                context.stroke(line, with: .color(Color(hex: spec.groups["structure"]?.color ?? "#8FA3B8")),
                               style: StrokeStyle(lineWidth: detailed ? 6 : 4, lineCap: .round))
            }
            for plane in map.planes where plane.vertical && (plane.kind == .door || plane.kind == .window) {
                let c = SIMD2(plane.center.x, plane.center.z)
                let q = toMap(c)
                context.fill(Path(ellipseIn: CGRect(x: q.x - 3, y: q.y - 3, width: 6, height: 6)),
                             with: .color(plane.kind == .door ? .green : .cyan))
            }

            if state.path.count > 1 {
                var walk = Path()
                walk.move(to: toMap(state.path[0]))
                for p in state.path.dropFirst() { walk.addLine(to: toMap(p)) }
                context.stroke(walk, with: .color(.cyan.opacity(0.9)), lineWidth: detailed ? 2 : 1.5)
            }

            // The phone and its view direction.
            let centre = toMap(state.position)
            let heading = Angle.degrees(state.heading - 90)
            var cone = Path()
            cone.move(to: centre)
            cone.addArc(center: centre, radius: detailed ? 40 : 18, startAngle: heading - .degrees(28),
                        endAngle: heading + .degrees(28), clockwise: false)
            cone.closeSubpath()
            context.fill(cone, with: .color(.yellow.opacity(0.4)))
            context.fill(Path(ellipseIn: CGRect(x: centre.x - 4, y: centre.y - 4, width: 8, height: 8)),
                         with: .color(.yellow))
        }
    }
}

/// The small map beside the camera, with the ring of directions filmed.
struct CoverageMap: View {
    @ObservedObject var state: CaptureState
    let spec: DetectionSpec

    var body: some View {
        ZStack {
            RoomMapCanvas(state: state, spec: spec)
                .clipShape(Circle())
            Canvas { context, size in
                let centre = CGPoint(x: size.width / 2, y: size.height / 2)
                let ringRadius = min(size.width, size.height) / 2 - 4
                for sector in 0..<12 {
                    let a0 = Angle.degrees(Double(sector) * 30 - 90)
                    let a1 = Angle.degrees(Double(sector + 1) * 30 - 90 - 3)
                    var arc = Path()
                    arc.addArc(center: centre, radius: ringRadius, startAngle: a0, endAngle: a1, clockwise: false)
                    context.stroke(arc, with: .color(state.sectors.contains(sector) ? .green : .white.opacity(0.18)),
                                   lineWidth: 4)
                }
            }
        }
        .background(.black.opacity(0.5), in: Circle())
        .overlay(Circle().stroke(.white.opacity(0.15)))
    }
}

/// Full-screen floor plan with a legend and what has been placed.
struct RoomMapSheet: View {
    @ObservedObject var state: CaptureState
    let spec: DetectionSpec
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                RoomMapCanvas(state: state, spec: spec, detailed: true)
                    .background(.black)
                    .frame(maxHeight: .infinity)
                List {
                    Section("Detected so far") {
                        let walls = state.map.walls.count, floors = state.map.floors.count
                        if walls + floors + state.map.objects.count == 0 {
                            Text("Nothing placed yet. Keep scanning; walls, the floor and furniture appear here as they are recognised.")
                                .foregroundStyle(.secondary)
                        }
                        if floors > 0 { Label("Floor", systemImage: "square.fill").foregroundStyle(Color(hex: "#8FA3B8")) }
                        if walls > 0 { Label("\(walls) wall\(walls == 1 ? "" : "s")", systemImage: "line.diagonal").foregroundStyle(Color(hex: "#8FA3B8")) }
                        ForEach(state.map.objects) { object in
                            HStack {
                                Circle().fill(Color(hex: spec.groups[object.group]?.color ?? "#FFFFFF")).frame(width: 10, height: 10)
                                Text(object.label.capitalized)
                                Spacer()
                                Text(String(format: "%.1f × %.1f × %.1f m", object.size.x, object.size.z, object.size.y))
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    Section("Legend") {
                        ForEach(["structure", "furniture", "storage", "soft", "fixture", "appliance"], id: \.self) { group in
                            HStack {
                                Circle().fill(Color(hex: spec.groups[group]?.color ?? "#FFFFFF")).frame(width: 10, height: 10)
                                Text(legend(group))
                            }
                        }
                    }
                }
                .frame(maxHeight: 300)
            }
            .navigationTitle("Room map")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }

    private func legend(_ group: String) -> String {
        switch group {
        case "structure": return "Walls, floor, doors, windows"
        case "furniture": return "Beds, tables, chairs, sofas"
        case "storage": return "Cupboards, wardrobes, shelves"
        case "soft": return "Curtains, rugs, cushions"
        case "fixture": return "Lamps, mirrors, pictures"
        case "appliance": return "TVs, screens, fridges"
        default: return group
        }
    }
}
