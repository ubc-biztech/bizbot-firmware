#!/usr/bin/env ruby
# Only needed after changing project membership; the generated project is committed.
require 'xcodeproj'
root = File.expand_path('..', __dir__)
project = Xcodeproj::Project.new(File.join(root, 'BizBot.xcodeproj'))
project.root_object.attributes['LastUpgradeCheck'] = '2640'
app = project.new_target(:application, 'BizBot', :ios, '18.0')
app.product_reference.name = 'BizBot.app'
group = project.main_group.new_group('App', 'App')
Dir.glob(File.join(root, 'App/**/*.swift')).sort.each do |path|
  app.source_build_phase.add_file_reference(group.new_file(path.delete_prefix(root + '/App/')))
end
resources = project.main_group.new_group('Shared', 'shared')
app.resources_build_phase.add_file_reference(resources.new_file('personalities.json'))
config = project.main_group.new_group('Config', 'Config')
config.new_file('Debug.plist')
config.new_file('Release.plist')
package = project.new(Xcodeproj::Project::Object::XCLocalSwiftPackageReference)
package.relative_path = '.'
project.root_object.package_references << package
dependency = project.new(Xcodeproj::Project::Object::XCSwiftPackageProductDependency)
dependency.product_name = 'BizBotCore'
dependency.package = package
app.package_product_dependencies << dependency
framework = project.new(Xcodeproj::Project::Object::PBXBuildFile)
framework.product_ref = dependency
app.frameworks_build_phase.files << framework
app.build_configurations.each do |configuration|
  settings = configuration.build_settings
  settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'dev.bizbot.interaction'
  settings['SWIFT_VERSION'] = '5.0'
  settings['TARGETED_DEVICE_FAMILY'] = '2'
  settings['IPHONEOS_DEPLOYMENT_TARGET'] = '18.0'
  settings['INFOPLIST_FILE'] = "Config/#{configuration.name}.plist"
  settings['GENERATE_INFOPLIST_FILE'] = 'NO'
  settings['CODE_SIGN_STYLE'] = 'Automatic'
  settings['SUPPORTED_PLATFORMS'] = 'iphoneos iphonesimulator'
  settings['SUPPORTS_MACCATALYST'] = 'NO'
  settings['SWIFT_EMIT_LOC_STRINGS'] = 'YES'
  settings['SWIFT_ACTIVE_COMPILATION_CONDITIONS'] = 'DEBUG' if configuration.name == 'Debug'
end
scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(app)
scheme.set_launch_target(app)
ui_tests = project.new_target(:ui_test_bundle, 'BizBotUITests', :ios, '18.0')
ui_tests.add_dependency(app)
test_group = project.main_group.new_group('UITests', 'UITests')
Dir.glob(File.join(root, 'UITests/*.swift')).sort.each do |path|
  ui_tests.source_build_phase.add_file_reference(test_group.new_file(File.basename(path)))
end
ui_tests.build_configurations.each do |configuration|
  configuration.build_settings.merge!({
    'PRODUCT_BUNDLE_IDENTIFIER' => 'dev.bizbot.interaction.uitests',
    'SWIFT_VERSION' => '5.0', 'TARGETED_DEVICE_FAMILY' => '2',
    'GENERATE_INFOPLIST_FILE' => 'YES', 'TEST_TARGET_NAME' => 'BizBot',
    'CODE_SIGN_STYLE' => 'Automatic'
  })
end
scheme.add_test_target(ui_tests)
scheme.save_as(File.join(root, 'BizBot.xcodeproj'), 'BizBot', true)
project.save
puts 'Generated BizBot.xcodeproj'
