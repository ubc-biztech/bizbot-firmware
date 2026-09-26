import Foundation

public enum FaceExpression: String, Codable, CaseIterable, Sendable {
    case neutral, happy, curious, thoughtful, surprised
}

public enum SessionPhase: String, Sendable {
    case idle, connecting, listening, thinking, speaking, reconnecting, disconnected
}

public struct Personality: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let instructions: String
    public let voice: String
    public let greetingsEnabled: Bool
    public let greetingPrompt: String
    public let arrivalSeconds: TimeInterval
    public let departureSeconds: TimeInterval
    public let cooldownSeconds: TimeInterval
}

public struct TrackedFace: Equatable, Sendable, Identifiable {
    public let id: UUID
    /// Normalized display coordinates: origin at top left, mirrored to match the display.
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double
    public init(id: UUID, x: Double, y: Double, width: Double, height: Double) {
        self.id = id; self.x = x; self.y = y; self.width = width; self.height = height
    }
    public var centerX: Double { x + width / 2 }
    public var centerY: Double { y + height / 2 }
    public var area: Double { width * height }
}

public struct CameraSnapshot: Sendable {
    public let jpeg: Data
    public let capturedAt: Date
    public init(jpeg: Data, capturedAt: Date) { self.jpeg = jpeg; self.capturedAt = capturedAt }
}

public struct AudioChunk: Sendable {
    public let responseID: String?
    public let itemID: String
    public let contentIndex: Int
    public let pcm: Data
    public init(itemID: String, contentIndex: Int, pcm: Data, responseID: String? = nil) {
        self.responseID = responseID
        self.itemID = itemID; self.contentIndex = contentIndex; self.pcm = pcm
    }
}

public struct PlaybackMark: Equatable, Sendable {
    public let itemID: String
    public let contentIndex: Int
    public let milliseconds: Int
    public init(itemID: String, contentIndex: Int, milliseconds: Int) {
        self.itemID = itemID; self.contentIndex = contentIndex; self.milliseconds = milliseconds
    }
}

public enum ConversationEvent: Sendable {
    case ready, userSpeechStarted, userSpeechEnded, responseFinished
    case responseStarted(id: String)
    case audio(AudioChunk)
    case expression(FaceExpression)
    case captureRequested(callID: String)
    case transcript(String)
    case failure(message: String, retryable: Bool)
}

@MainActor public protocol ConversationProvider: AnyObject {
    var onEvent: ((ConversationEvent) -> Void)? { get set }
    func connect(profile: Personality) async throws
    func sendAudio(_ pcm: Data)
    func sendSnapshot(_ snapshot: CameraSnapshot)
    func requestGreeting(_ instructions: String)
    func interrupt(at mark: PlaybackMark?)
    func completeCapture(callID: String, snapshot: CameraSnapshot?)
    func disconnect()
}

@MainActor public protocol PerceptionSource: AnyObject {
    var onFaces: (([TrackedFace], Date) -> Void)? { get set }
    func start() async throws
    func stop()
    func snapshot() async -> CameraSnapshot?
}

public enum InteractionAction: Equatable, Sendable {
    case greet(String)
}

public protocol InteractionPolicy {
    mutating func observe(faces: [TrackedFace], at time: Date, canSpeak: Bool) -> [InteractionAction]
    mutating func reset()
    mutating func resume(at time: Date)
}

/// Tokens prevent work from a previous connection or permission request affecting a new session.
public struct SessionGeneration: Sendable {
    public private(set) var value = UUID()
    public init() {}
    @discardableResult public mutating func advance() -> UUID { value = UUID(); return value }
    public func accepts(_ token: UUID) -> Bool { token == value }
}
