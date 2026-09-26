import SwiftUI
import AVFoundation
import BizBotCore

@MainActor final class AppModel: ObservableObject {
    @Published private(set) var phase: SessionPhase = .idle
    @Published private(set) var expression: FaceExpression = .neutral
    @Published private(set) var gaze = CGPoint(x: 0.5, y: 0.5)
    @Published private(set) var faceCount = 0
    @Published private(set) var transcript = ""
    @Published private(set) var status = "Ready when you are"
    @Published private(set) var preview: UIImage?
    @Published private(set) var active = false
    @Published private(set) var microphoneLevel: Double = 0
    @Published private(set) var sentAudioBytes = 0
    @Published private(set) var detectedSpeechTurns = 0
    private var lastMicrophoneUpdate = Date.distantPast
    @Published private(set) var sentImages = 0
    @Published var showsOperator = false
    @Published var showsPreview = false
    @Published var simulatedPerson = false
    let settings = AppSettings()

    private let camera = CameraService()
    private let audio = AudioService()
    private var provider: (any ConversationProvider)?
    private var policy: (any InteractionPolicy)?
    private var generation = SessionGeneration()
    private var startup: Task<Void, Never>?
    private var ticker: Task<Void, Never>?
    private var healthyTimer: Task<Void, Never>?
    private var expressionTimer: Task<Void, Never>?
    private var reconnectAttempts = 0
    private var liveReady = false
    private var playing = false
    private var microphoneResumeTime: TimeInterval = 0
    private var generating = false
    private var userSpeaking = false
    private var snapshotBusy = false
    private var lastSnapshot = Date.distantPast
    private var targetFace: UUID?
    private var lastFaceTime = Date.distantPast
    private var lastSpeechEnded = Date.distantPast
    private var activeMock = true
    private let mockFaceID = UUID()

    init() {
        camera.onFaces = { [weak self] faces, time in self?.observe(faces, at: time) }
        camera.onFailure = { [weak self] message in self?.fail(message, retryable: false) }
        audio.onInput = { [weak self] data in
            guard let self, self.active else { return }
            let now = Date()
            if now.timeIntervalSince(self.lastMicrophoneUpdate) >= 0.15 {
                let rms = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> Double in
                    let count = bytes.count / 2
                    guard count > 0 else { return 0 }
                    var energy = 0.0
                    for i in 0..<count {
                        let bits = UInt16(bytes[2 * i]) | (UInt16(bytes[2 * i + 1]) << 8)
                        let sample = Double(Int16(bitPattern: bits)) / 32768
                        energy += sample * sample
                    }
                    return sqrt(energy / Double(count))
                }
                self.microphoneLevel = rms > 0 ? max(0, min(1, (20 * log10(rms) + 60) / 60)) : 0
                self.lastMicrophoneUpdate = now
            }
            guard self.liveReady else { return }
            // Preserve the audio timeline for turn detection without transmitting
            // speaker audio on Mac while using ordinary microphone capture.
            let suppressEcho = self.audio.suppressMicrophoneDuringPlayback &&
                (self.playing || ProcessInfo.processInfo.systemUptime < self.microphoneResumeTime)
            self.provider?.sendAudio(suppressEcho ? Data(count: data.count) : data)
        }
        audio.onPlaybackChanged = { [weak self] playing in
            guard let self, self.active else { return }
            if self.playing && !playing {
                self.microphoneResumeTime = ProcessInfo.processInfo.systemUptime + 0.6
            }
            self.playing = playing; self.updatePhase()
        }
        audio.onFailure = { [weak self] message in self?.fail(message, retryable: false) }
    }

    func start() {
        guard !active else { return }
        guard let profile = settings.profile else { status = settings.configurationError ?? "Choose a personality."; return }
        do {
            if !settings.mockMode { _ = try settings.validatedURL(); try Keychain.save(settings.token) }
        } catch { status = error.localizedDescription; phase = .disconnected; return }
        active = true; activeMock = settings.mockMode; reconnectAttempts = 0
        policy = profile.greetingsEnabled ? GreetingPolicy(profile: profile) : PassivePolicy()
        transcript = ""; sentImages = 0; expression = .neutral
        sentAudioBytes = 0; detectedSpeechTurns = 0
        UIApplication.shared.isIdleTimerDisabled = true
        connect(profile: profile)
    }

    private func connect(profile: Personality, delay: Double = 0) {
        let token = generation.advance()
        liveReady = false; playing = false; generating = false; userSpeaking = false
        phase = delay > 0 ? .reconnecting : .connecting
        status = delay > 0 ? "Reconnecting (\(reconnectAttempts)/3)…" : "Starting \(activeMock ? "mock" : "live") session…"
        startup = Task { [weak self] in
            guard let self else { return }
            do {
                if delay > 0 { try await Task.sleep(for: .seconds(delay)) }
                guard self.generation.accepts(token), self.active else { return }
                let connection: any ConversationProvider
                if self.activeMock { connection = MockProvider() }
                else {
                    connection = OpenAIProvider(backendURL: try self.settings.validatedURL(), accessToken: self.settings.token)
                    (connection as? OpenAIProvider)?.onAudioSent = { [weak self] count in
                        guard let self, self.active, self.generation.accepts(token) else { return }
                        self.sentAudioBytes += count
                    }
                    try await self.camera.start()
                    guard self.generation.accepts(token), !Task.isCancelled else { return }
                    try await self.audio.start()
                }
                guard self.generation.accepts(token), !Task.isCancelled else { return }
                self.provider = connection
                connection.onEvent = { [weak self] event in
                    guard let self, self.active, self.generation.accepts(token) else { return }
                    self.handle(event)
                }
                try await connection.connect(profile: profile)
                guard self.generation.accepts(token), !Task.isCancelled else { return }
                self.startTicker(token: token)
            } catch is CancellationError {} catch {
                guard self.generation.accepts(token), self.active else { return }
                self.fail(error.localizedDescription, retryable: error is URLError)
            }
        }
    }

    func stop(message: String = "Session paused") {
        active = false; generation.advance()
        cleanup()
        policy?.reset(); policy = nil
        phase = .idle; expression = .neutral; status = message
        transcript = ""; preview = nil; faceCount = 0; gaze = CGPoint(x: 0.5, y: 0.5)
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func cleanup() {
        startup?.cancel(); ticker?.cancel(); healthyTimer?.cancel(); expressionTimer?.cancel()
        startup = nil; ticker = nil; healthyTimer = nil; expressionTimer = nil
        provider?.onEvent = nil; provider?.disconnect(); provider = nil
        camera.stop(); audio.stop()
        microphoneLevel = 0; lastMicrophoneUpdate = .distantPast; microphoneResumeTime = 0
        liveReady = false; playing = false; generating = false; userSpeaking = false; snapshotBusy = false
        lastSnapshot = .distantPast; lastFaceTime = .distantPast; targetFace = nil; preview = nil
    }

    private func fail(_ message: String, retryable: Bool) {
        guard active else { return }
        generation.advance(); cleanup()
        if retryable, reconnectAttempts < 3, let profile = settings.profile {
            reconnectAttempts += 1
            connect(profile: profile, delay: pow(2, Double(reconnectAttempts - 1)))
        } else {
            active = false; phase = .disconnected; status = message
            faceCount = 0; gaze = CGPoint(x: 0.5, y: 0.5)
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    private func handle(_ event: ConversationEvent) {
        switch event {
        case .ready:
            if reconnectAttempts > 0 { policy?.resume(at: Date()) }
            liveReady = true; status = activeMock ? "Mock session · camera and mic off" : "Live · camera and microphone on"
            updatePhase()
            let token = generation.value
            healthyTimer = Task { [weak self] in
                try? await Task.sleep(for: .seconds(30))
                guard let self, !Task.isCancelled, self.generation.accepts(token) else { return }
                self.reconnectAttempts = 0
            }
        case .userSpeechStarted:
            detectedSpeechTurns += 1
            userSpeaking = true; generating = false
            let mark = activeMock ? nil : audio.interrupt()
            provider?.interrupt(at: mark)
            playing = false; updatePhase()
        case .userSpeechEnded:
            userSpeaking = false; lastSpeechEnded = Date(); generating = true; updatePhase()
        case .responseStarted: generating = true; updatePhase()
        case .audio(let chunk):
            if activeMock { playing = true; updatePhase() } else { audio.play(chunk) }
        case .responseFinished:
            generating = false
            if activeMock { playing = false }
            updatePhase()
        case .expression(let value):
            expression = value
            expressionTimer?.cancel()
            let token = generation.value
            expressionTimer = Task { [weak self] in
                try? await Task.sleep(for: .seconds(6))
                guard let self, !Task.isCancelled, self.generation.accepts(token) else { return }
                self.expression = .neutral
            }
        case .captureRequested(let id):
            let token = generation.value
            Task { [weak self] in
                guard let self else { return }
                let snapshot = await self.camera.snapshot()
                guard self.active, self.generation.accepts(token) else { return }
                self.provider?.completeCapture(callID: id, snapshot: snapshot)
                if snapshot != nil { self.sentImages += 1; self.lastSnapshot = Date() }
            }
        case .transcript(let text): transcript = text
        case .failure(let message, let retryable): fail(message, retryable: retryable)
        }
    }

    private func updatePhase() {
        guard active, liveReady else { return }
        let next: SessionPhase = userSpeaking ? .listening : playing ? .speaking : generating ? .thinking : .listening
        if phase != next { phase = next }
    }

    private func observe(_ faces: [TrackedFace], at time: Date) {
        guard active else { return }
        if faceCount != faces.count { faceCount = faces.count }
        if let face = faces.first(where: { $0.id == targetFace }) ?? faces.max(by: { $0.area < $1.area }) {
            targetFace = face.id; lastFaceTime = time
            let next = CGPoint(x: face.centerX, y: face.centerY)
            if !showsOperator, gaze != next { gaze = next }
        } else if time.timeIntervalSince(lastFaceTime) > 1 {
            targetFace = nil
            let center = CGPoint(x: 0.5, y: 0.5)
            if gaze != center { gaze = center }
        }
        let canSpeak = liveReady && !userSpeaking && !playing && !generating && time.timeIntervalSince(lastSpeechEnded) > 2
        let actions = policy?.observe(faces: faces, at: time, canSpeak: canSpeak) ?? []
        for action in actions {
            switch action {
            case .greet(let instructions):
                generating = true; updatePhase(); provider?.requestGreeting(instructions)
            }
        }
    }

    private func startTicker(token: UUID) {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.active, self.generation.accepts(token) else { return }
                if self.activeMock {
                    let face = TrackedFace(id: self.mockFaceID, x: 0.35 + sin(Date().timeIntervalSince1970 / 3) * 0.12,
                                               y: 0.3, width: 0.25, height: 0.35)
                    self.observe(self.simulatedPerson ? [face] : [], at: Date())
                } else if self.liveReady {
                    let send = Date().timeIntervalSince(self.lastSnapshot) >= 5
                    if (send || self.showsPreview) && !self.snapshotBusy {
                        self.snapshotBusy = true
                        let snapshot = await self.camera.snapshot()
                        guard self.generation.accepts(token), !Task.isCancelled else { return }
                        self.snapshotBusy = false
                        if let snapshot {
                            if self.showsPreview { self.preview = UIImage(data: snapshot.jpeg) }
                            if send { self.provider?.sendSnapshot(snapshot); self.sentImages += 1; self.lastSnapshot = Date() }
                        }
                    }
                }
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
    }

    func simulateConversation() { (provider as? MockProvider)?.simulateConversation() }
    func saveSettings() {
        do { try Keychain.save(settings.token) }
        catch { status = error.localizedDescription }
    }
}
