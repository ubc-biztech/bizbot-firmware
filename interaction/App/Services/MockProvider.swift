import Foundation
import BizBotCore

@MainActor final class MockProvider: ConversationProvider {
    var onEvent: ((ConversationEvent) -> Void)?
    private var responseTask: Task<Void, Never>?
    private var connected = false

    func connect(profile: Personality) async throws { connected = true; onEvent?(.ready) }
    func sendAudio(_ pcm: Data) {}
    func sendSnapshot(_ snapshot: CameraSnapshot) {}
    func requestGreeting(_ instructions: String) { reply("Hello! I'm BizBot. What would you like to talk about?") }
    func interrupt(at mark: PlaybackMark?) { responseTask?.cancel(); onEvent?(.responseFinished) }
    func completeCapture(callID: String, snapshot: CameraSnapshot?) {}
    func disconnect() { connected = false; responseTask?.cancel(); responseTask = nil }

    func simulateConversation() {
        guard connected else { return }
        onEvent?(.userSpeechStarted)
        reply("I can keep you company and describe what’s in front of me. This is a simulated response.")
    }
    private func reply(_ text: String) {
        responseTask?.cancel()
        responseTask = Task { [weak self] in
            guard let self, self.connected else { return }
            self.onEvent?(.userSpeechEnded)
            self.onEvent?(.responseStarted(id: UUID().uuidString))
            try? await Task.sleep(for: .milliseconds(650))
            guard !Task.isCancelled, self.connected else { return }
            self.onEvent?(.expression(.happy)); self.onEvent?(.transcript(text))
            // Silent mock audio exercises the same playback/state path without recording anything.
            self.onEvent?(.audio(AudioChunk(itemID: UUID().uuidString, contentIndex: 0, pcm: Data(count: 96_000))))
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled, self.connected else { return }
            self.onEvent?(.responseFinished)
        }
    }
}
