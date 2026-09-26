import Foundation

public struct GreetingPolicy: InteractionPolicy {
    private struct Presence {
        var firstSeen: Date
        var lastSeen: Date
        var greeted = false
    }
    private let profile: Personality
    private var people: [UUID: Presence] = [:]
    private var lastGreeting: Date?
    private var resumeUntil: Date?

    public init(profile: Personality) { self.profile = profile }
    public mutating func reset() { people.removeAll(); lastGreeting = nil; resumeUntil = nil }

    /// After reconnecting, treat already-present people as greeted instead of replaying an arrival.
    public mutating func resume(at time: Date) {
        people.removeAll()
        resumeUntil = time.addingTimeInterval(3)
    }

    public mutating func observe(faces: [TrackedFace], at time: Date, canSpeak: Bool) -> [InteractionAction] {
        people = people.filter { time.timeIntervalSince($0.value.lastSeen) < profile.departureSeconds }
        for face in faces {
            if var person = people[face.id] {
                // A brief occlusion must not count as a full second of stable initial presence.
                if !person.greeted && time.timeIntervalSince(person.lastSeen) > 0.6 {
                    person.firstSeen = time
                }
                person.lastSeen = time
                people[face.id] = person
            } else {
                people[face.id] = Presence(firstSeen: time, lastSeen: time)
            }
            if let resumeUntil, time < resumeUntil { people[face.id]?.greeted = true }
        }
        guard profile.greetingsEnabled, canSpeak,
              lastGreeting.map({ time.timeIntervalSince($0) >= profile.cooldownSeconds }) ?? true else { return [] }
        let eligible = faces.filter { face in
            guard let person = people[face.id] else { return false }
            return !person.greeted && time.timeIntervalSince(person.firstSeen) >= profile.arrivalSeconds
        }
        guard !eligible.isEmpty else { return [] }
        // A single greeting addresses the currently present group.
        for face in faces { people[face.id]?.greeted = true }
        lastGreeting = time
        return [.greet(profile.greetingPrompt)]
    }
}

public struct PassivePolicy: InteractionPolicy {
    public init() {}
    public mutating func reset() {}
    public mutating func resume(at time: Date) {}
    public mutating func observe(faces: [TrackedFace], at time: Date, canSpeak: Bool) -> [InteractionAction] { [] }
}

/// Spatial tracks are ephemeral; they are never biometric identities.
public struct FaceTracker {
    private struct Track { var face: TrackedFace; var seen: Date }
    private var tracks: [UUID: Track] = [:]
    public init() {}
    public mutating func reset() { tracks.removeAll() }
    public mutating func update(rects: [(x: Double, y: Double, width: Double, height: Double)], at time: Date) -> [TrackedFace] {
        tracks = tracks.filter { time.timeIntervalSince($0.value.seen) < 10 }
        var available = Set(tracks.keys)
        return rects.sorted { $0.width * $0.height > $1.width * $1.height }.map { rect in
            let cx = rect.x + rect.width / 2, cy = rect.y + rect.height / 2
            let match = available.min { a, b in
                hypot(tracks[a]!.face.centerX - cx, tracks[a]!.face.centerY - cy) <
                hypot(tracks[b]!.face.centerX - cx, tracks[b]!.face.centerY - cy)
            }
            let id: UUID
            if let match, let track = tracks[match], hypot(track.face.centerX - cx, track.face.centerY - cy) < 0.22 {
                id = match; available.remove(match)
            } else { id = UUID() }
            let face = TrackedFace(id: id, x: rect.x, y: rect.y, width: rect.width, height: rect.height)
            tracks[id] = Track(face: face, seen: time)
            return face
        }
    }
}
