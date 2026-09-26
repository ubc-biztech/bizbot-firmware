# BizBot interaction

A native iPad face, local person tracking, and hands-free cloud conversation. The app is independent of the ESP32 firmware and sends no motor commands.

## Try the face without credentials

1. Open `BizBot.xcodeproj` in Xcode 16 or newer (built here with Xcode 26.4).
2. Choose the **BizBot** scheme and an iPad simulator running iPadOS 18 or newer.
3. Run the app and tap **Start session**. The first launch defaults to **Mock mode**.
4. Open the sliders button in the upper-right corner. Tap **Add simulated person**, then close the controls. BizBot looks toward the simulated person and greets them once.
5. Use **Simulate a conversation turn** to exercise thinking, expression, and speaking states. The mock displays captions without playing sound or accessing the microphone/camera.
6. Pause, select **Quiet companion**, and restart to try the passive interaction policy.

For a physical iPad, select your signing team in the app target, change the bundle identifier if necessary, select the connected iPad, and run. The app supports both landscape orientations, iPadOS 18+, and the built-in front camera, microphone, and speakers.

## Run a live conversation

The Mac and iPad must be on the same **internet-connected** Wi-Fi. The firmware's `BizBot-Control` access point has no internet and is not used by this version. The Mac must stay running; it issues session credentials while media flows directly between the iPad and OpenAI.

From this folder:

```sh
cd server
npm ci
cp .env.example .env
npm run token
```

Edit `server/.env`: set `OPENAI_API_KEY` to your API key and `BIZBOT_ACCESS_TOKEN` to the generated token. Keep these values out of source control. `OPENAI_REALTIME_MODEL` defaults to `gpt-realtime`; your API project must have access to the configured model.

```sh
npm run build
npm start
```

Find the Mac's local hostname in **System Settings → General → Sharing**, or run:

```sh
scutil --get LocalHostName
```

On the iPad, open **BizBot controls**, pause any running session, and:

- Disable **Mock mode**.
- Set the backend URL to `http://YOUR-MAC-HOSTNAME.local:8787`.
- Enter the **backend access token**, not your OpenAI API key, and save it to Keychain.
- Start the session and allow Local Network, Camera, and Microphone access.

The app uses ordinary microphone capture without echo cancellation. On Mac, microphone transmission pauses during playback and for 0.6 seconds afterward to prevent speaker feedback; wait for BizBot to finish before replying. Ask a visual question such as “What am I holding?” to request a fresh camera image. The operator panel has a camera preview, current face count, image count, and the last generated response. Camera snapshots are intermittent, not continuous video understanding.

The Debug build allows HTTP only for local hostnames; this is for development on your trusted Wi-Fi. Release builds require an HTTPS backend. No internet-facing server is deployed by this project. The backend is authenticated and limits credential issuance to ten requests per minute for this single-iPad prototype.

## Behavior and modularity

`Sources/BizBotCore` is a portable Swift package with no camera, audio, SwiftUI, or networking dependencies:

| Interface | Responsibility |
| --- | --- |
| `PerceptionSource` | Start/stop perception; produce temporary face observations and timestamped snapshots. |
| `ConversationProvider` | Connect, stream PCM, send images, handle interruption, and emit typed conversation events. |
| `InteractionPolicy` | Turn presence and conversation availability into interaction actions. |
| `FaceExpression` / `SessionPhase` | Describe appearance separately from animation and provider events. |

The app's composition point is `AppModel`. `OpenAIProvider` owns OpenAI wire messages; `MockProvider` exercises the same event interface. A new provider can implement `ConversationProvider` and be selected there without changing the face renderer or camera service. `RealtimeCodec` is isolated OpenAI-specific parsing code inside the otherwise provider-neutral core package.

Edit `shared/personalities.json` to change tone, voice, greeting instructions, or timing. The app bundles this file and the backend reads the same file at startup. **Rebuild the app and restart the backend after editing it.** Profile IDs must match between them. No separate sentiment classifier is hard-coded: the conversation provider can emit expression events, while the interaction policy determines when to initiate speech. `GreetingPolicy` and `PassivePolicy` are the initial replaceable implementations.

Defaults:

- Local face detection at approximately 5 Hz, with spatial tracking and no biometric identification.
- One second of stable presence before greeting; ten seconds of absence before a track expires; thirty seconds between greetings. One greeting addresses the current group. Speech and playback defer greetings.
- Reconnect suppresses greeting people already in view during its first three seconds, preventing old arrival events from being replayed.
- One JPEG every five seconds, plus fresh images requested through `capture_scene`. Maximum dimension 768 pixels; only two image messages are retained in the cloud conversation at a time.
- English, the `marin` voice, and a concise friendly personality. `set_expression` accepts neutral, happy, curious, thoughtful, or surprised expressions.
- Three reconnect attempts with 1/2/4-second delays. Thirty healthy seconds reset the retry budget. Reconnect creates a fresh cloud conversation rather than replaying old audio or images.
- Foreground sessions only. Pause/backgrounding stops capture and playback, disconnects the provider, clears the local transcript/preview, and restores normal display sleep.

The native audio service converts input to 24 kHz mono PCM16, uses unprocessed microphone capture, and tracks actual played samples. On interruption it clears playback and tells the provider where to truncate the response. WebSockets keep the prototype free of third-party native dependencies; a future WebRTC transport belongs inside a replacement provider/audio adapter.

## Backend interface

`GET /health` returns `{ "status": "ok" }` without credentials.

`POST /session` requires `Authorization: Bearer <BIZBOT_ACCESS_TOKEN>` and JSON:

```json
{ "profileId": "friendly-greeter" }
```

Success returns:

```json
{ "clientSecret": "short-lived-provider-token", "expiresAt": 1234567890, "model": "gpt-realtime" }
```

The backend selects the configured model, instructions, audio formats, and tools; clients cannot override them in the request. Responses use `Cache-Control: no-store`. The permanent API key stays on the Mac. Errors return a small error code without the provider's body or credentials. The app stores only its backend token in Keychain and ordinary preferences in UserDefaults.

Neither the app nor the backend writes conversations, audio, or images to disk. Live media is sent to OpenAI and remains subject to the provider's data handling policies. Transcripts and camera previews shown in the app are session-only. Faces are approximate spatial tracks and may be confused by crossings or long occlusions.

## Verification

```sh
swift test
cd server
npm test
```

Build the simulator app from the project root:

```sh
xcodebuild -project BizBot.xcodeproj -scheme BizBot \
  -sdk iphonesimulator -configuration Debug \
  -derivedDataPath /tmp/bizbot-ipad-build CODE_SIGNING_ALLOWED=NO build
```

Run the UI test using any installed iPad simulator's name or UUID:

```sh
xcodebuild -project BizBot.xcodeproj -scheme BizBot \
  -destination 'platform=iOS Simulator,name=iPad Pro 11-inch (M5)' \
  -derivedDataPath /tmp/bizbot-ipad-build CODE_SIGNING_ALLOWED=NO test
```

Core tests cover presence debounce, deferral, cooldown, reconnect, tracker expiry, played-audio truncation, bounded image context, stale-session tokens, and provider parsing. Backend tests use a mocked upstream and check authentication, profile selection, malformed/oversized requests, expired credentials, upstream failure, secret handling, and issuance limits. The UI test checks mock greeting, pause, personality switching, a conversation turn, and backgrounding.

The checked-in Xcode project opens directly. If adding app files or changing target membership, regenerate it with `ruby scripts/generate_project.rb` (requires the Ruby `xcodeproj` gem). The app has no external Swift packages; `BizBotCore` is local.

### Physical iPad acceptance checklist

These checks require a real iPad, working API credentials, and the Mac backend:

- Check camera orientation, text legibility in scene images, and correct eye direction in both landscape orientations.
- Approach, remain present, leave briefly, and return after ten seconds; verify greetings do not repeat continuously. Try two people and an occlusion.
- Hold up an object and ask what it is; ensure the answer uses a fresh image.
- Interrupt a long reply. Confirm playback stops promptly, the robot hears the interruption, and it does not answer its own speaker output.
- Deny each permission, then enable it in Settings and restart. Confirm permission prompts themselves do not cancel startup.
- Disconnect Wi-Fi, stop the backend, and test an incorrect token. Confirm clear status, bounded retries, and no stale speech or greetings after reconnecting.
- Background and reopen the app. Confirm the session remains paused until explicitly started.
- Run a thirty-minute live session and inspect responsiveness, memory, battery/heat, gaze stability, and repeated greetings. Simulator tests cannot validate speaker feedback or camera performance.

## References

- [OpenAI Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [OpenAI client secrets](https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create)
- [Apple face detection](https://developer.apple.com/documentation/vision/vndetectfacerectanglesrequest)
- [Apple voice processing](https://developer.apple.com/documentation/avfaudio/avaudioionode/setvoiceprocessingenabled(_:))
