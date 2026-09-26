import AVFoundation
import BizBotCore

@MainActor final class AudioService {
    var onInput: ((Data) -> Void)?
    var onPlaybackChanged: ((Bool) -> Void)?
    var onFailure: ((String) -> Void)?
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private(set) var suppressMicrophoneDuringPlayback = false
    private let playbackFormat = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
    private var ledger = PlaybackLedger()
    private var generation = SessionGeneration()
    private var playbackGeneration = SessionGeneration()
    private var running = false
    private var pendingBuffers = 0
    private var completedFrames: Int64 = 0
    private var itemOffsets: [String: Int] = [:]
    private var lastCompletedMark: PlaybackMark?
    private var observers: [NSObjectProtocol] = []

    init() {
        engine.attach(player)
        observers = [AVAudioSession.interruptionNotification, AVAudioSession.routeChangeNotification,
                     AVAudioSession.mediaServicesWereResetNotification].map { name in
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] notification in
                Task { @MainActor in
                    guard let self, self.running else { return }
                    if notification.name == AVAudioSession.routeChangeNotification {
                        let raw = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt ?? 0
                        let reason = AVAudioSession.RouteChangeReason(rawValue: raw)
                        guard reason == .newDeviceAvailable || reason == .oldDeviceUnavailable || reason == .noSuitableRouteForCategory else { return }
                    }
                    if notification.name == AVAudioSession.interruptionNotification,
                       let type = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                       type != AVAudioSession.InterruptionType.began.rawValue { return }
                    self.onFailure?("The audio device changed or was interrupted. Start the session again.")
                }
            }
        }
    }

    func start() async throws {
        let token = generation.advance()
        let allowed = await AVAudioApplication.requestRecordPermission()
        guard generation.accepts(token), !Task.isCancelled else { throw CancellationError() }
        guard allowed else { throw AppFailure.message("Microphone access is off. Enable it for BizBot in Settings.") }
        let audioSession = AVAudioSession.sharedInstance()
        let runningOnMac = ProcessInfo.processInfo.isiOSAppOnMac
        try audioSession.setCategory(.playAndRecord, mode: .default,
                                     options: runningOnMac ? [] : [.defaultToSpeaker])
        try audioSession.setPreferredIOBufferDuration(0.02)
        try audioSession.setActive(true)
        let input = engine.inputNode
        engine.disconnectNodeOutput(input)
        engine.disconnectNodeOutput(player)
        engine.disconnectNodeOutput(engine.mainMixerNode)
        try input.setVoiceProcessingEnabled(false)
        suppressMicrophoneDuringPlayback = runningOnMac
        let hardwareFormat = input.outputFormat(forBus: 0)
        guard hardwareFormat.sampleRate > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24_000, channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: hardwareFormat, to: target) else {
            throw AppFailure.message("The microphone format is unavailable.")
        }
        engine.connect(player, to: engine.mainMixerNode, format: playbackFormat)
        engine.connect(engine.mainMixerNode, to: engine.outputNode, format: nil)
        converter.downmix = true
        // Conversion occurs on the audio callback, with only copied PCM crossing to the main actor.
        input.installTap(onBus: 0, bufferSize: 2048, format: hardwareFormat) { [weak self] buffer, _ in
            let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * 24_000 / hardwareFormat.sampleRate)) + 32
            guard let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var supplied = false
            var error: NSError?
            let conversionStatus = converter.convert(to: converted, error: &error) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true; status.pointee = .haveData; return buffer
            }
            guard error == nil, conversionStatus != .error else {
                Task { @MainActor in
                    guard let self, self.running, self.generation.accepts(token) else { return }
                    self.onFailure?("Microphone audio conversion failed. Check the Mac input device and restart the session.")
                }
                return
            }
            guard let samples = converted.int16ChannelData?[0], converted.frameLength > 0 else { return }
            let data = Data(bytes: samples, count: Int(converted.frameLength) * 2)
            Task { @MainActor in
                guard let self, self.running, self.generation.accepts(token) else { return }
                self.onInput?(data)
            }
        }
        running = true
        do { engine.prepare(); try engine.start() }
        catch { stop(); throw error }
    }

    func play(_ chunk: AudioChunk) {
        guard running, chunk.pcm.count >= 2 else { return }
        let frames = chunk.pcm.count / 2
        // Bound playback to 20 seconds instead of allowing an unlimited audio backlog.
        guard ledger.scheduledFrames - playedFrames() + Int64(frames) <= 24_000 * 20 else {
            onFailure?("The voice playback buffer filled. Start a new session."); return
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: playbackFormat, frameCapacity: AVAudioFrameCount(frames)),
              let samples = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = AVAudioFrameCount(frames)
        chunk.pcm.withUnsafeBytes { raw in
            for i in 0..<frames {
                let value = UInt16(raw[i * 2]) | (UInt16(raw[i * 2 + 1]) << 8)
                samples[i] = Float(Int16(bitPattern: value)) / 32768
            }
        }
        ledger.append(itemID: chunk.itemID, contentIndex: chunk.contentIndex, frames: Int64(frames))
        pendingBuffers += 1
        let token = generation.value
        let playbackToken = playbackGeneration.value
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation.accepts(token), self.playbackGeneration.accepts(playbackToken), self.running else { return }
                self.completedFrames += Int64(frames)
                self.pendingBuffers -= 1
                if self.pendingBuffers == 0 {
                    self.lastCompletedMark = self.adjustedMark(self.ledger.mark(at: self.completedFrames))
                    if let mark = self.lastCompletedMark { self.itemOffsets = [mark.itemID: mark.milliseconds] }
                    self.player.stop(); self.ledger.reset(); self.completedFrames = 0
                    self.onPlaybackChanged?(false)
                }
            }
        }
        if !player.isPlaying { player.play() }
        onPlaybackChanged?(true)
    }

    func interrupt() -> PlaybackMark? {
        let mark = adjustedMark(ledger.mark(at: playedFrames())) ?? lastCompletedMark
        // Invalidate queued playback callbacks while keeping the microphone token valid.
        playbackGeneration.advance()
        player.stop()
        ledger.reset(); pendingBuffers = 0; completedFrames = 0
        itemOffsets.removeAll(); lastCompletedMark = nil
        onPlaybackChanged?(false)
        return mark
    }

    private func adjustedMark(_ mark: PlaybackMark?) -> PlaybackMark? {
        guard let mark else { return nil }
        return PlaybackMark(itemID: mark.itemID, contentIndex: mark.contentIndex,
                            milliseconds: mark.milliseconds + (itemOffsets[mark.itemID] ?? 0))
    }

    private func playedFrames() -> Int64 {
        guard let time = player.lastRenderTime, let position = player.playerTime(forNodeTime: time) else { return completedFrames }
        return max(completedFrames, min(ledger.scheduledFrames, position.sampleTime))
    }

    func stop() {
        generation.advance()
        playbackGeneration.advance()
        let wasRunning = running
        running = false
        player.stop(); engine.stop()
        if wasRunning { engine.inputNode.removeTap(onBus: 0) }
        ledger.reset(); pendingBuffers = 0; completedFrames = 0; itemOffsets.removeAll(); lastCompletedMark = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    deinit { observers.forEach(NotificationCenter.default.removeObserver) }
}
