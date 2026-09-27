#pragma once

// NVS survives power loss and normal firmware uploads (not full flash erase).
float loadPitchOffset(float fallback);
bool savePitchOffset(float offset);
