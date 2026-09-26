import Foundation

/// This codec is specific to the OpenAI adapter. The rest of the app uses ConversationEvent.
public enum RealtimeMessage {
    case event(ConversationEvent)
    case tool(name: String, callID: String, arguments: Data)
    case done(hasTools: Bool, failed: Bool, cancelled: Bool)
    case ignored
}

public enum RealtimeCodec {
    public static func decode(_ data: Data) throws -> RealtimeMessage {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return .ignored }
        switch type {
        case "session.created": return .event(.ready)
        case "input_audio_buffer.speech_started": return .event(.userSpeechStarted)
        case "input_audio_buffer.speech_stopped": return .event(.userSpeechEnded)
        case "response.created":
            let response = json["response"] as? [String: Any]
            return .event(.responseStarted(id: response?["id"] as? String ?? ""))
        case "response.output_audio.delta":
            guard let item = json["item_id"] as? String,
                  let encoded = json["delta"] as? String, let pcm = Data(base64Encoded: encoded), pcm.count % 2 == 0 else { return .ignored }
            return .event(.audio(AudioChunk(itemID: item, contentIndex: json["content_index"] as? Int ?? 0, pcm: pcm, responseID: json["response_id"] as? String)))
        case "response.output_audio_transcript.done":
            return .event(.transcript(json["transcript"] as? String ?? ""))
        case "response.function_call_arguments.done":
            guard let name = json["name"] as? String, let id = json["call_id"] as? String,
                  let arguments = json["arguments"] as? String else { return .ignored }
            return .tool(name: name, callID: id, arguments: Data(arguments.utf8))
        case "response.done":
            let response = json["response"] as? [String: Any] ?? [:]
            let output = response["output"] as? [[String: Any]] ?? []
            return .done(hasTools: output.contains { $0["type"] as? String == "function_call" },
                         failed: response["status"] as? String == "failed", cancelled: response["status"] as? String == "cancelled")
        case "error":
            let error = json["error"] as? [String: Any] ?? [:]
            let code = error["code"] as? String ?? "unknown"
            // Cancellation can race with completion; neither requires restarting the session.
            if code == "response_cancel_not_active" { return .ignored }
            let retryable = ["server_error", "session_expired"].contains(code)
            return .event(.failure(message: "Voice service error (\(code)).", retryable: retryable))
        default: return .ignored
        }
    }
}

public struct ImageContextWindow {
    private var ids: [String] = []
    private let limit: Int
    public init(limit: Int = 2) { self.limit = max(1, limit) }
    /// Returns items that must be deleted before adding this image.
    public mutating func insert(_ id: String) -> [String] {
        ids.append(id)
        let count = max(0, ids.count - limit)
        let removed = Array(ids.prefix(count))
        ids.removeFirst(count)
        return removed
    }
    public mutating func reset() { ids.removeAll() }
}
