#include "OffsetStorage.h"
#include <Preferences.h>
#include <math.h>

namespace {
constexpr const char* NAMESPACE = "bizbot-cal";
constexpr const char* KEY = "pitch-trim";
bool valid(float value) { return isfinite(value) && fabsf(value) <= 15.0f; }
}

float loadPitchOffset(float fallback) {
    Preferences storage;
    if (!storage.begin(NAMESPACE, true)) return fallback;
    const float value = storage.getFloat(KEY, fallback);
    storage.end();
    return valid(value) ? value : fallback;
}

bool savePitchOffset(float offset) {
    if (!valid(offset)) return false;
    Preferences storage;
    if (!storage.begin(NAMESPACE, false)) return false;
    // Avoid wearing flash when the requested offset hasn't changed.
    const bool saved = storage.getFloat(KEY, NAN) == offset ||
                       storage.putFloat(KEY, offset) == sizeof(float);
    storage.end();
    return saved;
}
