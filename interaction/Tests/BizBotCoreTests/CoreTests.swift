import XCTest
@testable import BizBotCore

final class CoreTests: XCTestCase {
    private let epoch = Date(timeIntervalSince1970: 1_000)
    private func profile() throws -> Personality {
        try JSONDecoder().decode(Personality.self, from: Data("""
        {"id":"test","name":"Test","instructions":"Be concise","voice":"marin","greetingsEnabled":true,
        "greetingPrompt":"Hello","arrivalSeconds":1,"departureSeconds":10,"cooldownSeconds":30}
        """.utf8))
    }
    private func face(_ id: UUID = UUID()) -> TrackedFace { TrackedFace(id: id, x: 0.3, y: 0.2, width: 0.2, height: 0.3) }

    func testGreetingDebounceDeferralAndNoRepeat() throws {
        var policy = GreetingPolicy(profile: try profile())
        let person = face()
        for tick in 0..<6 { XCTAssertTrue(policy.observe(faces: [person], at: epoch.addingTimeInterval(Double(tick) * 0.2), canSpeak: false).isEmpty) }
        XCTAssertEqual(policy.observe(faces: [person], at: epoch.addingTimeInterval(1.2), canSpeak: true), [.greet("Hello")])
        XCTAssertTrue(policy.observe(faces: [person], at: epoch.addingTimeInterval(2), canSpeak: true).isEmpty)
        // Keep observing so this remains the same arrival, even after the cooldown.
        for second in 3...40 { XCTAssertTrue(policy.observe(faces: [person], at: epoch.addingTimeInterval(Double(second)), canSpeak: true).isEmpty) }
    }

    func testDepartureCooldownAndNewArrival() throws {
        var policy = GreetingPolicy(profile: try profile())
        let first = face(), second = face()
        for tick in 0...5 { _ = policy.observe(faces: [first], at: epoch.addingTimeInterval(Double(tick) * 0.2), canSpeak: true) }
        for tick in 10...20 { XCTAssertTrue(policy.observe(faces: [second], at: epoch.addingTimeInterval(Double(tick) * 0.2), canSpeak: true).isEmpty) }
        _ = policy.observe(faces: [], at: epoch.addingTimeInterval(20), canSpeak: true)
        var results: [InteractionAction] = []
        for tick in 0...6 { results += policy.observe(faces: [second], at: epoch.addingTimeInterval(32 + Double(tick) * 0.2), canSpeak: true) }
        XCTAssertEqual(results, [.greet("Hello")])
    }

    func testOcclusionDoesNotCountAsStablePresenceAndPassiveNeverGreets() throws {
        var policy = GreetingPolicy(profile: try profile())
        let person = face()
        _ = policy.observe(faces: [person], at: epoch, canSpeak: true)
        XCTAssertTrue(policy.observe(faces: [person], at: epoch.addingTimeInterval(2), canSpeak: true).isEmpty)
        var passive: any InteractionPolicy = PassivePolicy()
        for tick in 0...100 { XCTAssertTrue(passive.observe(faces: [person], at: epoch.addingTimeInterval(Double(tick)), canSpeak: true).isEmpty) }
    }

    func testTrackerHandlesOcclusionAndExpiresWithoutIdentityRecognition() {
        var tracker = FaceTracker()
        let initial = tracker.update(rects: [(0.2, 0.2, 0.2, 0.2)], at: epoch)
        let moved = tracker.update(rects: [(0.25, 0.2, 0.2, 0.2)], at: epoch.addingTimeInterval(0.2))
        XCTAssertEqual(initial[0].id, moved[0].id)
        let group = tracker.update(rects: [(0.25, 0.2, 0.2, 0.2), (0.7, 0.2, 0.2, 0.2)], at: epoch.addingTimeInterval(0.4))
        XCTAssertEqual(Set(group.map(\.id)).count, 2)
        let returned = tracker.update(rects: [(0.25, 0.2, 0.2, 0.2)], at: epoch.addingTimeInterval(12))
        XCTAssertNotEqual(initial[0].id, returned[0].id)
    }

    func testReconnectDoesNotReplayAnExistingArrival() throws {
        var policy = GreetingPolicy(profile: try profile())
        let existing = face(), newcomer = face()
        policy.resume(at: epoch)
        for tick in 0...25 {
            XCTAssertTrue(policy.observe(faces: [existing], at: epoch.addingTimeInterval(Double(tick) * 0.2), canSpeak: true).isEmpty)
        }
        var actions: [InteractionAction] = []
        for tick in 26...32 {
            actions += policy.observe(faces: [existing, newcomer], at: epoch.addingTimeInterval(Double(tick) * 0.2), canSpeak: true)
        }
        XCTAssertEqual(actions, [.greet("Hello")])
    }

    func testPlaybackTruncationTracksPlayedSamplesInsteadOfReceivedAudio() {
        var ledger = PlaybackLedger()
        ledger.append(itemID: "a", contentIndex: 0, frames: 24_000)
        ledger.append(itemID: "a", contentIndex: 0, frames: 24_000)
        XCTAssertEqual(ledger.mark(at: 12_000), PlaybackMark(itemID: "a", contentIndex: 0, milliseconds: 500))
        ledger.append(itemID: "b", contentIndex: 0, frames: 24_000)
        XCTAssertEqual(ledger.mark(at: 60_000), PlaybackMark(itemID: "b", contentIndex: 0, milliseconds: 500))
        XCTAssertEqual(ledger.mark(at: 100_000)?.milliseconds, 1000)
        ledger.reset(); XCTAssertNil(ledger.mark(at: 1))
    }

    func testImageContextAndSessionGenerationAreBounded() {
        var images = ImageContextWindow(limit: 2)
        XCTAssertEqual(images.insert("a"), []); XCTAssertEqual(images.insert("b"), [])
        XCTAssertEqual(images.insert("c"), ["a"])
        images.reset(); XCTAssertEqual(images.insert("d"), [])
        var generation = SessionGeneration()
        let old = generation.value
        XCTAssertTrue(generation.accepts(old))
        let fresh = generation.advance()
        XCTAssertFalse(generation.accepts(old)); XCTAssertTrue(generation.accepts(fresh))
    }

    func testRealtimeParsingAndMalformedAudio() throws {
        let pcm = Data([0, 0, 1, 0])
        let message = Data("""
        {"type":"response.output_audio.delta","item_id":"a","content_index":0,"delta":"\(pcm.base64EncodedString())"}
        """.utf8)
        guard case .event(.audio(let chunk)) = try RealtimeCodec.decode(message) else { return XCTFail("Expected audio") }
        XCTAssertEqual(chunk.pcm, pcm); XCTAssertEqual(chunk.itemID, "a")
        guard case .ignored = try RealtimeCodec.decode(Data("{\"type\":\"response.output_audio.delta\",\"delta\":\"bad\"}".utf8)) else { return XCTFail("Expected ignored malformed audio") }
        guard case .tool(let name, let id, _) = try RealtimeCodec.decode(Data("""
        {"type":"response.function_call_arguments.done","name":"capture_scene","call_id":"tool1","arguments":"{}"}
        """.utf8)) else { return XCTFail("Expected tool") }
        XCTAssertEqual(name, "capture_scene"); XCTAssertEqual(id, "tool1")
        guard case .ignored = try RealtimeCodec.decode(Data("{\"type\":\"future.event\"}".utf8)) else { return XCTFail("Unknown events should be ignored") }
        guard case .done(_, false, true) = try RealtimeCodec.decode(Data("{\"type\":\"response.done\",\"response\":{\"status\":\"cancelled\",\"output\":[]}}".utf8)) else { return XCTFail("Cancelled responses must not continue tools") }
        guard case .event(.failure(_, false)) = try RealtimeCodec.decode(Data("{\"type\":\"error\",\"error\":{\"code\":\"invalid_api_key\"}}".utf8)) else { return XCTFail("Credential failures must not loop") }
        guard case .event(.failure(_, true)) = try RealtimeCodec.decode(Data("{\"type\":\"error\",\"error\":{\"code\":\"session_expired\"}}".utf8)) else { return XCTFail("Expired sessions can reconnect") }
    }
}
