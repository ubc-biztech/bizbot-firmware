#pragma once
// Minimal host shim for the pure control classes; not used in ESP32 builds.
#include <cmath>
template <typename T> T constrain(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}
