import SwiftUI

@main struct BizBotApp: App {
    @StateObject private var model = AppModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            FaceScreen(model: model)
                .preferredColorScheme(.dark)
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background, model.active { model.stop(message: "Paused while the app was away") }
                }
        }
    }
}
