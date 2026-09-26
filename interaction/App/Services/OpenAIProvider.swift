import Foundation
import BizBotCore

@MainActor final class OpenAIProvider: ConversationProvider {
    var onAudioSent: ((Int) -> Void)?
    var onEvent: ((ConversationEvent) -> Void)?
    private let backendURL: URL
    private let accessToken: String
    private var socket: URLSessionWebSocketTask?
    private var session: URLSession?
    private var receiveTask: Task<Void, Never>?
    private var sendTask: Task<Void, Never>?
    private var watchdog: Task<Void, Never>?
    private var generation = SessionGeneration()
    private var queue: [(data: Data, audioBytes: Int)] = []
    private var queuedBytes = 0
    private var images = ImageContextWindow()
    private var pendingTools: Set<String> = []
    private var continueAfterTools = false
    private var ready = false
    private var responseActive = false
    private var interruptedItems: Set<String> = []
    private var activeResponseID: String?
    private var interruptedResponseID: String?
    private var userSpeaking = false
    private var personalityInstructions = ""

    init(backendURL: URL, accessToken: String) { self.backendURL = backendURL; self.accessToken = accessToken }

    func connect(profile: Personality) async throws {
        disconnect()
        personalityInstructions = profile.instructions
        let token = generation.value
        var request = URLRequest(url: backendURL.appendingPathComponent("session"))
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["profileId": profile.id])
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 20
        let network = URLSession(configuration: configuration)
        session = network
        let (data, response) = try await network.data(for: request)
        guard generation.accepts(token) else { throw CancellationError() }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            let message = status == 401 ? "Backend token was rejected. Check the operator settings." :
                status == 400 ? "The backend does not have this personality profile. Restart it after updating profiles." :
                status == 429 ? "Voice service is rate limited. Wait a minute and start again." : "Backend could not create a voice session (HTTP \(status))."
            throw AppFailure.message(message)
        }
        struct Credentials: Decodable { let clientSecret: String; let expiresAt: Double; let model: String }
        let credentials = try JSONDecoder().decode(Credentials.self, from: data)
        guard credentials.expiresAt > Date().timeIntervalSince1970 + 5 else { throw AppFailure.message("Session credential expired. Start again.") }
        var endpoint = URLComponents(string: "wss://api.openai.com/v1/realtime")!
        endpoint.queryItems = [URLQueryItem(name: "model", value: credentials.model)]
        var upgrade = URLRequest(url: endpoint.url!)
        upgrade.setValue("Bearer \(credentials.clientSecret)", forHTTPHeaderField: "Authorization")
        let connection = network.webSocketTask(with: upgrade)
        connection.maximumMessageSize = 2 * 1024 * 1024
        socket = connection
        connection.resume()
        receiveTask = Task { [weak self] in
            do {
                while !Task.isCancelled {
                    let message = try await connection.receive()
                    guard let self, self.generation.accepts(token) else { return }
                    let payload: Data
                    switch message {
                    case .data(let data): payload = data
                    case .string(let string): payload = Data(string.utf8)
                    @unknown default: continue
                    }
                    self.handle(try RealtimeCodec.decode(payload))
                }
            } catch {
                guard let self, self.generation.accepts(token), !Task.isCancelled else { return }
                self.fail("Voice connection was lost.", retryable: true)
            }
        }
        watchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(15))
            guard let self, !Task.isCancelled, self.generation.accepts(token), !self.ready else { return }
            self.fail("Voice service did not become ready.", retryable: true)
        }
    }

    func sendAudio(_ pcm: Data) {
        guard ready else { return }
        enqueue(["type": "input_audio_buffer.append", "audio": pcm.base64EncodedString()], audioBytes: pcm.count)
    }

    func sendSnapshot(_ snapshot: CameraSnapshot) {
        guard ready, Date().timeIntervalSince(snapshot.capturedAt) < 2, snapshot.jpeg.count < 350_000 else { return }
        let id = "img_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
        for removed in images.insert(id) { enqueue(["type": "conversation.item.delete", "item_id": removed]) }
        enqueue(["type": "conversation.item.create", "item": [
            "id": id, "type": "message", "role": "user",
            "content": [
                ["type": "input_text", "text": "Silent camera observation at \(ISO8601DateFormatter().string(from: snapshot.capturedAt)). Use as context; do not speak just because this image arrived."],
                ["type": "input_image", "image_url": "data:image/jpeg;base64," + snapshot.jpeg.base64EncodedString()]
            ]
        ]])
    }

    func requestGreeting(_ instructions: String) {
        guard ready, !responseActive else { return }
        responseActive = true
        enqueue(["type": "response.create", "response": ["instructions": personalityInstructions + "\nFor this response: " + instructions]])
    }

    func interrupt(at mark: PlaybackMark?) {
        // VAD cancels generation server-side. We own playback and its accurate truncation.
        if let mark {
            interruptedItems.insert(mark.itemID)
            enqueue(["type": "conversation.item.truncate", "item_id": mark.itemID,
                     "content_index": mark.contentIndex, "audio_end_ms": mark.milliseconds])
        }
        continueAfterTools = false
    }

    func completeCapture(callID: String, snapshot: CameraSnapshot?) {
        guard pendingTools.contains(callID) else { return }
        if let snapshot { sendSnapshot(snapshot) }
        toolResult(callID, value: snapshot == nil ? "Camera unavailable. Explain that you cannot currently see." : "A fresh camera image has been added to the conversation.")
        finishToolsIfReady()
    }

    func disconnect() {
        generation.advance()
        receiveTask?.cancel(); sendTask?.cancel(); watchdog?.cancel()
        receiveTask = nil; sendTask = nil; watchdog = nil
        socket?.cancel(with: .goingAway, reason: nil); socket = nil
        session?.invalidateAndCancel(); session = nil
        ready = false; responseActive = false
        queue.removeAll(); queuedBytes = 0; images.reset()
        pendingTools.removeAll(); continueAfterTools = false; interruptedItems.removeAll()
        activeResponseID = nil; interruptedResponseID = nil; userSpeaking = false
    }

    private func handle(_ message: RealtimeMessage) {
        switch message {
        case .event(let event):
            switch event {
            case .ready: ready = true; watchdog?.cancel()
            case .responseStarted(let id):
                responseActive = true; activeResponseID = id; interruptedItems.removeAll()
            case .userSpeechStarted:
                userSpeaking = true; interruptedResponseID = activeResponseID; continueAfterTools = false
            case .userSpeechEnded: userSpeaking = false
            case .audio(let chunk):
                if interruptedItems.contains(chunk.itemID) || userSpeaking ||
                    (chunk.responseID != nil && chunk.responseID == interruptedResponseID) { return }
            case .failure(let message, let retryable): fail(message, retryable: retryable); return
            default: break
            }
            onEvent?(event)
        case .tool(let name, let id, let arguments):
            pendingTools.insert(id)
            if userSpeaking || (activeResponseID != nil && activeResponseID == interruptedResponseID) {
                toolResult(id, value: "Action cancelled by user interruption."); return
            }
            if name == "set_expression" {
                let body = (try? JSONSerialization.jsonObject(with: arguments)) as? [String: String]
                if let raw = body?["expression"], let expression = FaceExpression(rawValue: raw) {
                    onEvent?(.expression(expression)); toolResult(id, value: "Expression updated.")
                } else { toolResult(id, value: "Unsupported expression.") }
            } else if name == "capture_scene" { onEvent?(.captureRequested(callID: id)) }
            else { toolResult(id, value: "Unsupported tool.") }
        case .done(let hasTools, let failed, let cancelled):
            responseActive = false
            if failed { fail("The voice response failed. Start a new session.", retryable: false); return }
            continueAfterTools = hasTools && !cancelled && !userSpeaking && activeResponseID != interruptedResponseID
            if continueAfterTools { finishToolsIfReady() } else { onEvent?(.responseFinished) }
        case .ignored: break
        }
    }

    private func toolResult(_ id: String, value: String) {
        enqueue(["type": "conversation.item.create", "item": ["type": "function_call_output", "call_id": id, "output": value]])
        pendingTools.remove(id)
    }

    private func finishToolsIfReady() {
        guard continueAfterTools, pendingTools.isEmpty, ready else { return }
        continueAfterTools = false; responseActive = true
        enqueue(["type": "response.create"])
    }

    private func enqueue(_ object: [String: Any], audioBytes: Int = 0) {
        guard let socket, let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        // Never accumulate seconds of old audio on a stalled uplink.
        guard queue.count < 40, queuedBytes + data.count < 600_000 else {
            fail("Network is too slow for live audio. Reconnecting…", retryable: true); return
        }
        queue.append((data, audioBytes)); queuedBytes += data.count
        guard sendTask == nil else { return }
        let token = generation.value
        sendTask = Task { [weak self] in
            guard let self else { return }
            do {
                while !self.queue.isEmpty && !Task.isCancelled && self.generation.accepts(token) {
                    let packet = self.queue.removeFirst(); self.queuedBytes -= packet.data.count
                    try await socket.send(.string(String(decoding: packet.data, as: UTF8.self)))
                    guard self.generation.accepts(token) else { return }
                    if packet.audioBytes > 0 { self.onAudioSent?(packet.audioBytes) }
                }
                if self.generation.accepts(token) { self.sendTask = nil }
            } catch {
                guard self.generation.accepts(token), !Task.isCancelled else { return }
                self.fail("Could not send live audio.", retryable: true)
            }
        }
    }

    private func fail(_ message: String, retryable: Bool) {
        disconnect()
        onEvent?(.failure(message: message, retryable: retryable))
    }
}
