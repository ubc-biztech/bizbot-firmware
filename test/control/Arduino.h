#pragma once
// Minimal host shim for the pure control classes; not used in ESP32 builds.
#include <cmath>
template <typename T> T constrain(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

// Host-only Arduino interfaces used by command-parser regression tests.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
unsigned long millis();
class Print {
public:
    std::string text;
    template <typename T> void print(T value) {
        std::ostringstream stream;
        stream << value;
        text += stream.str();
    }
    void print(float value, int digits) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(digits) << value;
        text += stream.str();
    }
    template <typename T> void println(T value) { print(value); text += '\n'; }
    void println(float value, int digits) { print(value, digits); text += '\n'; }
};
class Stream : public Print {
public:
    virtual int available() = 0;
    virtual int read() = 0;
};
