import Foundation

/// Maps the audio player's sample cursor back to the provider's conversation item.
public struct PlaybackLedger {
    private struct Segment {
        let itemID: String
        let contentIndex: Int
        let start: Int64
        var frames: Int64
    }
    private var segments: [Segment] = []
    public private(set) var scheduledFrames: Int64 = 0
    public init() {}
    public mutating func reset() { segments.removeAll(); scheduledFrames = 0 }
    public mutating func append(itemID: String, contentIndex: Int, frames: Int64) {
        guard frames > 0 else { return }
        if let last = segments.last, last.itemID == itemID && last.contentIndex == contentIndex {
            segments[segments.count - 1].frames += frames
        } else {
            segments.append(Segment(itemID: itemID, contentIndex: contentIndex, start: scheduledFrames, frames: frames))
        }
        scheduledFrames += frames
    }
    public func mark(at playedFrames: Int64) -> PlaybackMark? {
        let cursor = max(0, min(playedFrames, scheduledFrames))
        guard let segment = segments.last(where: { $0.start <= cursor }) else { return nil }
        let played = min(segment.frames, cursor - segment.start)
        return PlaybackMark(itemID: segment.itemID, contentIndex: segment.contentIndex, milliseconds: Int(played * 1000 / 24_000))
    }
}
