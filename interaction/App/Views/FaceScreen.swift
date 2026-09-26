import SwiftUI
import BizBotCore

struct FaceScreen: View {
    @ObservedObject var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private var testing: Bool { ProcessInfo.processInfo.arguments.contains("--ui-testing") }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.ignoresSafeArea()
                TimelineView(.animation(minimumInterval: reduceMotion ? 0.25 : 1.0 / 60, paused: testing)) { timeline in
                    let t = timeline.date.timeIntervalSinceReferenceDate
                    let cycle = t.truncatingRemainder(dividingBy: 5.7)
                    let blink = reduceMotion ? 1 : cycle < 0.16 ? max(0.08, abs(cycle - 0.08) / 0.08) : 1
                    let breath = reduceMotion ? 1 : 1 + sin(t * 1.7) * 0.015
                    HStack(spacing: geometry.size.width * 0.13) {
                        eye(isLeft: true, time: t, blink: blink, breath: breath, size: geometry.size)
                        eye(isLeft: false, time: t, blink: blink, breath: breath, size: geometry.size)
                    }
                    .offset(x: (model.gaze.x - 0.5) * geometry.size.width * 0.13,
                            y: (model.gaze.y - 0.5) * geometry.size.height * 0.13)
                    .animation(.easeOut(duration: 0.24), value: model.gaze)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("BizBot is \(model.phase.rawValue), with a \(model.expression.rawValue) expression")
                }
                VStack {
                    HStack {
                        Text("B I Z B O T")
                            .font(.system(size: 12, weight: .medium, design: .rounded)).foregroundStyle(.white.opacity(0.3))
                        Spacer()
                        Button { model.showsOperator = true } label: {
                            Image(systemName: "slider.horizontal.3").font(.system(size: 19)).frame(width: 48, height: 48)
                        }
                        .foregroundStyle(.white.opacity(0.5)).accessibilityLabel("Operator settings")
                    }
                    Spacer()
                    if model.settings.mockMode, !model.transcript.isEmpty {
                        Text(model.transcript).font(.system(size: 18, design: .rounded))
                            .foregroundStyle(.white.opacity(0.65)).multilineTextAlignment(.center)
                            .frame(maxWidth: 650).padding(.bottom, 12)
                    }
                    HStack(spacing: 10) {
                        Circle().fill(model.active ? Color.cyan : Color.gray).frame(width: 5, height: 5)
                        Text(model.status).font(.system(size: 12, design: .rounded)).foregroundStyle(.white.opacity(0.5))
                        Spacer()
                        Button { model.active ? model.stop() : model.start() } label: {
                            Label(model.active ? "Pause" : "Start session", systemImage: model.active ? "pause.fill" : "play.fill")
                                .font(.system(size: 14, weight: .medium, design: .rounded))
                                .padding(.horizontal, 18).padding(.vertical, 14)
                                .background(.white.opacity(0.08), in: Capsule())
                                .contentShape(Capsule())
                        }
                        .buttonStyle(.plain).foregroundStyle(.white.opacity(0.8)).accessibilityIdentifier("sessionControl")
                    }
                }
                .padding(.horizontal, 32).padding(.vertical, 20)
            }
        }
        .persistentSystemOverlays(.hidden)
        .statusBarHidden(true)
        .sheet(isPresented: $model.showsOperator) { OperatorPanel(model: model, settings: model.settings) }
    }

    private func eye(isLeft: Bool, time: Double, blink: Double, breath: Double, size: CGSize) -> some View {
        let expression = model.phase == .thinking ? FaceExpression.thoughtful : model.expression
        let baseHeight = size.height * 0.31
        let height = expression == .happy ? baseHeight * 0.57 : expression == .thoughtful ? baseHeight * 0.65 : baseHeight
        let width = size.width * 0.165
        let speaking = model.phase == .speaking && !reduceMotion ? 1 + sin(time * 11) * 0.035 : 1
        let tilt: Double = expression == .curious ? (isLeft ? -7 : 7) : 0
        return RoundedRectangle(cornerRadius: min(width, height) * (expression == .surprised ? 0.5 : 0.34), style: .continuous)
            .fill(LinearGradient(colors: [Color(red: 0.5, green: 0.95, blue: 1), Color(red: 0.1, green: 0.72, blue: 0.88)], startPoint: .top, endPoint: .bottom))
            .frame(width: width, height: height)
            .scaleEffect(x: breath, y: blink * breath * speaking)
            .rotationEffect(.degrees(tilt))
            .opacity(model.phase == .disconnected ? 0.3 : model.active ? 1 : 0.7)
            .shadow(color: .cyan.opacity(0.17), radius: 28)
            .animation(.spring(response: 0.4, dampingFraction: 0.8), value: expression)
    }
}
