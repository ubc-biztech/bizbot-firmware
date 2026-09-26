import Foundation
import Combine
import Security
import BizBotCore

enum AppFailure: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let text) = self { return text }; return nil }
}

enum Keychain {
    private static let service = "dev.bizbot.interaction"
    static func read() -> String {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecAttrAccount as String: "backend-token",
            kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess, let data = result as? Data else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }
    static func save(_ token: String) throws {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecAttrAccount as String: "backend-token"]
        SecItemDelete(query as CFDictionary)
        guard !token.isEmpty else { return }
        var attributes = query
        attributes[kSecValueData as String] = Data(token.utf8)
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        guard SecItemAdd(attributes as CFDictionary, nil) == errSecSuccess else {
            throw AppFailure.message("Could not save the backend token in Keychain.")
        }
    }
}

@MainActor final class AppSettings: ObservableObject {
    @Published var backendURL: String { didSet { UserDefaults.standard.set(backendURL, forKey: "backendURL") } }
    @Published var profileID: String { didSet { UserDefaults.standard.set(profileID, forKey: "profileID") } }
    @Published var mockMode: Bool { didSet { UserDefaults.standard.set(mockMode, forKey: "mockMode") } }
    @Published var token: String
    let profiles: [Personality]
    let configurationError: String?

    init() {
        let testing = ProcessInfo.processInfo.arguments.contains("--ui-testing")
        backendURL = UserDefaults.standard.string(forKey: "backendURL") ?? "http://your-mac.local:8787"
        profileID = testing ? "friendly-greeter" : UserDefaults.standard.string(forKey: "profileID") ?? "friendly-greeter"
        mockMode = testing ? true : UserDefaults.standard.object(forKey: "mockMode") as? Bool ?? true
        token = Keychain.read()
        do {
            guard let url = Bundle.main.url(forResource: "personalities", withExtension: "json") else {
                throw AppFailure.message("Personality profiles are missing from the app bundle.")
            }
            profiles = try JSONDecoder().decode([Personality].self, from: Data(contentsOf: url))
            configurationError = profiles.isEmpty ? "No personality profiles were configured." : nil
        } catch { profiles = []; configurationError = "Could not load personality profiles. Rebuild the app with shared/personalities.json." }
    }

    var profile: Personality? { profiles.first { $0.id == profileID } ?? profiles.first }

    func validatedURL() throws -> URL {
        guard let url = URL(string: backendURL.trimmingCharacters(in: .whitespacesAndNewlines)),
              let host = url.host, url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil,
              url.path.isEmpty || url.path == "/" else {
            throw AppFailure.message("Enter the backend's base URL, such as http://your-mac.local:8787.")
        }
        if url.scheme == "https" { return url }
        #if DEBUG
        if url.scheme == "http" && (host.hasSuffix(".local") || host == "localhost") { return url }
        #endif
        throw AppFailure.message("Use HTTPS, or a .local Mac hostname over HTTP in a Debug build.")
    }
}
