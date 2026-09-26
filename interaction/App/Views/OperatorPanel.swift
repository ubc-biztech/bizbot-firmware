import SwiftUI

struct OperatorPanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Session") {
                    LabeledContent("State", value: model.phase.rawValue.capitalized)
                    Text(model.status).foregroundStyle(.secondary)
                    Button(model.active ? "Pause session" : "Start session") {
                        model.active ? model.stop() : model.start()
                    }
                }
                if settings.mockMode {
                    Section("Mock controls · no camera or microphone") {
                        Button {
                            model.simulatedPerson.toggle()
                        } label: {
                            Text(model.simulatedPerson ? "Remove simulated person" : "Add simulated person")
                                .frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
                        }
                        .accessibilityIdentifier("simulatePresence")
                        Button("Simulate a conversation turn") { model.simulateConversation() }.disabled(!model.active)
                        Text("The mock shows captions and speaking animation without sound.").foregroundStyle(.secondary)
                    }
                } else {
                    Section("Microphone") {
                        if ProcessInfo.processInfo.isiOSAppOnMac {
                            Text("Microphone transmission pauses during replies and resumes shortly afterward. Wait for BizBot to finish before replying.")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                        ProgressView(value: model.microphoneLevel)
                            .accessibilityLabel("Microphone input level")
                        LabeledContent("Audio sent", value: String(format: "%.1f seconds", Double(model.sentAudioBytes) / 48_000))
                        LabeledContent("Speech turns detected", value: String(model.detectedSpeechTurns))
                        Text("The meter measures audio captured by BizBot. Speech turns count when the voice service detects you speaking.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    Section("Camera") {
                        Toggle("Show camera preview", isOn: $model.showsPreview)
                        if model.showsPreview, let preview = model.preview {
                            Image(uiImage: preview).resizable().scaledToFit().frame(maxHeight: 240)
                        }
                        LabeledContent("People visible", value: String(model.faceCount))
                        LabeledContent("Images sent this session", value: String(model.sentImages))
                    }
                }

                Section {
                    Toggle("Mock mode", isOn: $settings.mockMode)
                    Picker("Personality", selection: $settings.profileID) {
                        ForEach(settings.profiles) { profile in Text(profile.name).tag(profile.id) }
                    }
                    .accessibilityIdentifier("personalityPicker")
                    TextField("http://your-mac.local:8787", text: $settings.backendURL)
                        .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                    SecureField("Backend access token", text: $settings.token)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    Button("Save token to Keychain") { model.saveSettings() }
                } header: { Text("Connection and personality") }
                footer: { Text("Pause the session before changing settings. Live mode sends microphone audio and periodic camera images to OpenAI. The Mac stores the permanent API key.") }
                .disabled(model.active)

                if !model.transcript.isEmpty {
                    Section("Last response · session only") { Text(model.transcript).textSelection(.enabled) }
                }
                Section {
                    Text("BizBot’s voice is AI-generated. This prototype does not control the robot’s motors or save recordings. Keep this app in the foreground while interacting.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Link("Open app permissions", destination: URL(string: UIApplication.openSettingsURLString)!)
                }
            }
            .navigationTitle("BizBot controls")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { model.saveSettings(); dismiss() } } }
        }
        .onDisappear { model.showsPreview = false }
    }
}
