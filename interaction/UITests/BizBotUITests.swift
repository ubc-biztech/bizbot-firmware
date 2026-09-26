import XCTest

final class BizBotUITests: XCTestCase {
    @MainActor func testMockGreetingPauseAndPersonalitySwap() throws {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchArguments = ["--ui-testing"]
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()
        func openControls() {
            let panel = app.navigationBars["BizBot controls"]
            for _ in 0..<2 {
                if panel.exists { return }
                app.buttons["Operator settings"].press(forDuration: 0.15)
                if panel.waitForExistence(timeout: 3) { return }
            }
            XCTFail("Operator controls did not open")
        }
        func reveal(_ element: XCUIElement) {
            for _ in 0..<4 {
                let form = app.collectionViews.firstMatch
                if element.exists {
                    let top = app.navigationBars["BizBot controls"].frame.maxY + 4
                    let bottom = form.frame.maxY - 24
                    if element.isHittable && element.frame.minY >= top && element.frame.maxY <= bottom { return }
                    if element.frame.minY < top { form.swipeDown(); continue }
                }
                form.swipeUp()
            }
        }
        let session = app.buttons["sessionControl"]
        XCTAssertTrue(session.waitForExistence(timeout: 10))
        func startSession() {
            let status = app.staticTexts["Mock session · camera and mic off"]
            // Simulator input can be dropped immediately after launch/rotation; retry only if still idle.
            for _ in 0..<2 {
                if session.label == "Start session" { session.tap() }
                if status.waitForExistence(timeout: 5) { return }
            }
            XCTFail("Mock session did not start")
        }
        startSession()
        openControls()
        let person = app.buttons["simulatePresence"]
        reveal(person)
        person.tap()
        XCTAssertTrue(app.buttons["Remove simulated person"].waitForExistence(timeout: 3))
        app.buttons["Done"].tap()
        let greeting = app.staticTexts["Hello! I'm BizBot. What would you like to talk about?"]
        XCTAssertTrue(greeting.waitForExistence(timeout: 5))
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "BizBot mock greeting"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        session.tap()
        XCTAssertTrue(app.staticTexts["Session paused"].waitForExistence(timeout: 3))
        XCTAssertFalse(greeting.exists)
        openControls()
        reveal(app.buttons["personalityPicker"])
        app.buttons["personalityPicker"].coordinate(withNormalizedOffset: CGVector(dx: 0.8, dy: 0.5)).tap()
        app.buttons["Quiet companion"].tap()
        app.buttons["Done"].tap()
        startSession()
        XCTAssertFalse(greeting.waitForExistence(timeout: 3))
        openControls()
        reveal(app.buttons["Simulate a conversation turn"])
        app.buttons["Simulate a conversation turn"].tap()
        app.buttons["Done"].tap()
        XCTAssertTrue(app.staticTexts["I can keep you company and describe what’s in front of me. This is a simulated response."].waitForExistence(timeout: 4))
        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertTrue(app.staticTexts["Paused while the app was away"].waitForExistence(timeout: 5))
    }
}
