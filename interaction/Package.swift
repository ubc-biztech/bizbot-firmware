// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "BizBotCore",
    platforms: [.macOS(.v13), .iOS("18.0")],
    products: [.library(name: "BizBotCore", targets: ["BizBotCore"])],
    targets: [
        .target(name: "BizBotCore"),
        .testTarget(name: "BizBotCoreTests", dependencies: ["BizBotCore"])
    ]
)
