import AVFoundation
import Vision
import CoreImage
import UIKit
import BizBotCore

@MainActor final class CameraService: PerceptionSource {
    var onFaces: (([TrackedFace], Date) -> Void)?
    var onFailure: ((String) -> Void)?
    private let worker = CaptureWorker()
    private var generation = SessionGeneration()
    private var running = false
    private var acceptsFramesAfter = Date.distantFuture

    init() {
        worker.onFaces = { [weak self] faces, time in
            Task { @MainActor in
                guard let self, self.running, time >= self.acceptsFramesAfter else { return }
                self.onFaces?(faces, time)
            }
        }
        worker.onFailure = { [weak self] in
            Task { @MainActor in
                guard let self, self.running else { return }
                self.onFailure?("Camera was interrupted. Stop and start the session to try again.")
            }
        }
    }

    func start() async throws {
        let token = generation.advance()
        let allowed = await AVCaptureDevice.requestAccess(for: .video)
        guard generation.accepts(token), !Task.isCancelled else { throw CancellationError() }
        guard allowed else { throw AppFailure.message("Camera access is off. Enable it for BizBot in Settings.") }
        running = true
        acceptsFramesAfter = Date()
        do { try await worker.start() }
        catch { running = false; throw error }
        guard generation.accepts(token), !Task.isCancelled else { worker.stop(); throw CancellationError() }
    }

    func stop() { generation.advance(); running = false; worker.stop() }
    func snapshot() async -> CameraSnapshot? {
        guard running else { return nil }
        let token = generation.value
        let frame = await worker.snapshot()
        return generation.accepts(token) ? frame : nil
    }
}

/// All capture state lives on this serial queue, never on the UI or audio thread.
private final class CaptureWorker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    var onFaces: (([TrackedFace], Date) -> Void)?
    var onFailure: (() -> Void)?
    private let queue = DispatchQueue(label: "dev.bizbot.camera", qos: .userInitiated)
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var configured = false
    private var buffer: CVPixelBuffer?
    private var capturedAt = Date.distantPast
    private var lastDetection = Date.distantPast
    private var tracker = FaceTracker()
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?
    private var observers: [NSObjectProtocol] = []

    func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    if !self.configured { try self.configure() }
                    self.tracker.reset(); self.buffer = nil
                    self.session.startRunning()
                    guard self.session.isRunning else { throw AppFailure.message("The front camera could not start.") }
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    private func configure() throws {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw AppFailure.message("No front camera is available. Use Mock mode in the Simulator.")
        }
        let input = try AVCaptureDeviceInput(device: device)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .vga640x480
        guard session.canAddInput(input), session.canAddOutput(output) else { throw AppFailure.message("Camera configuration is unavailable.") }
        session.addInput(input)
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(output)
        let rotation = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
        rotationCoordinator = rotation
        rotationObservation = rotation.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]) { [weak self] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelCapture
            self?.queue.async { [weak self] in self?.applyRotation(angle) }
        }
        if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        observers = [AVCaptureSession.runtimeErrorNotification, AVCaptureSession.wasInterruptedNotification].map { name in
            NotificationCenter.default.addObserver(forName: name, object: session, queue: nil) { [weak self] _ in self?.onFailure?() }
        }
        configured = true
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
            self.buffer = nil; self.tracker.reset()
        }
    }

    private func applyRotation(_ angle: CGFloat) {
        guard let connection = output.connection(with: .video) else { return }
        if connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
            buffer = nil
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let time = Date()
        buffer = pixelBuffer; capturedAt = time
        guard time.timeIntervalSince(lastDetection) >= 0.2 else { return }
        lastDetection = time
        let request = VNDetectFaceRectanglesRequest()
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
            let rects = (request.results ?? []).map { face in
                let b = face.boundingBox
                return (x: Double(1 - b.maxX), y: Double(1 - b.maxY), width: Double(b.width), height: Double(b.height))
            }
            onFaces?(tracker.update(rects: rects, at: time), time)
        } catch { onFaces?([], time) }
    }

    func snapshot() async -> CameraSnapshot? {
        await withCheckedContinuation { continuation in
            queue.async {
                guard let buffer = self.buffer, Date().timeIntervalSince(self.capturedAt) < 1 else { continuation.resume(returning: nil); return }
                var image = CIImage(cvPixelBuffer: buffer)
                let scale = min(1, 768 / max(image.extent.width, image.extent.height))
                image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                guard let jpeg = self.context.jpegRepresentation(of: image, colorSpace: CGColorSpaceCreateDeviceRGB(),
                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.65]) else {
                    continuation.resume(returning: nil); return
                }
                continuation.resume(returning: CameraSnapshot(jpeg: jpeg, capturedAt: self.capturedAt))
            }
        }
    }

    deinit { observers.forEach(NotificationCenter.default.removeObserver) }
}
