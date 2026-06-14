// UTF-8 utility functions for safe streaming and JSON serialization.
//
// Extracted from sse_emitter.cpp so that unit tests can validate these
// independently without constructing a full SseEmitter.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dflash::common {

// Snap a byte offset back to a UTF-8 code-point boundary.
// Returns the largest position <= `pos` that doesn't split a multi-byte sequence.
// (Mirrors ds4_server.c's utf8_stream_safe_len.)
inline size_t utf8_safe_len(const std::string & s, size_t pos) {
    if (pos >= s.size()) return s.size();
    while (pos > 0 && (s[pos] & 0xC0) == 0x80) pos--;
    return pos;
}

// Length of the prefix of `s` that ends on a complete UTF-8 codepoint.
// If `s` ends mid-sequence (a lead byte without all its continuation
// bytes yet), returns the offset of that dangling lead byte so the caller
// can hold the incomplete tail back for the next chunk. A complete (or
// invalid-lead) final sequence yields s.size(). Used to reassemble
// codepoints split across streaming token boundaries.
inline size_t utf8_complete_len(const std::string & s) {
    if (s.empty()) return 0;
    size_t i = s.size() - 1;
    // Walk back over continuation bytes to the lead byte of the last seq.
    while (i > 0 && ((uint8_t)s[i] & 0xC0) == 0x80) i--;
    uint8_t c = (uint8_t)s[i];
    size_t need;
    if (c < 0x80) need = 1;
    else if ((c & 0xE0) == 0xC0) need = 2;
    else if ((c & 0xF0) == 0xE0) need = 3;
    else if ((c & 0xF8) == 0xF0) need = 4;
    else return s.size();  // invalid lead — don't hold back; sanitize fixes it
    const size_t have = s.size() - i;
    if (have >= need) return s.size();  // last sequence is complete
    return i;                           // incomplete tail: hold back from i
}

// Sanitize a string for JSON: replace invalid/incomplete UTF-8 with U+FFFD.
inline std::string utf8_sanitize(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        uint8_t c = (uint8_t)s[i];
        size_t seq_len = 0;
        if (c < 0x80) seq_len = 1;
        else if ((c & 0xE0) == 0xC0) seq_len = 2;
        else if ((c & 0xF0) == 0xE0) seq_len = 3;
        else if ((c & 0xF8) == 0xF0) seq_len = 4;
        if (seq_len == 0 || i + seq_len > s.size()) {
            out += "\xEF\xBF\xBD";  // U+FFFD
            i++;
            continue;
        }
        bool valid = true;
        for (size_t j = 1; j < seq_len; j++) {
            if (((uint8_t)s[i + j] & 0xC0) != 0x80) { valid = false; break; }
        }
        if (valid) {
            // Decode codepoint and validate range.
            uint32_t cp = 0;
            if (seq_len == 1) {
                cp = c;
            } else if (seq_len == 2) {
                cp = ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)((uint8_t)s[i+1]) & 0x3F);
                if (cp < 0x80) valid = false;  // overlong
            } else if (seq_len == 3) {
                cp = ((uint32_t)(c & 0x0F) << 12) |
                     ((uint32_t)((uint8_t)s[i+1] & 0x3F) << 6) |
                     ((uint32_t)((uint8_t)s[i+2]) & 0x3F);
                if (cp < 0x800) valid = false;  // overlong
                if (cp >= 0xD800 && cp <= 0xDFFF) valid = false;  // surrogate
            } else {
                cp = ((uint32_t)(c & 0x07) << 18) |
                     ((uint32_t)((uint8_t)s[i+1] & 0x3F) << 12) |
                     ((uint32_t)((uint8_t)s[i+2] & 0x3F) << 6) |
                     ((uint32_t)((uint8_t)s[i+3]) & 0x3F);
                if (cp < 0x10000 || cp > 0x10FFFF) valid = false;  // overlong or out-of-range
            }
        }
        if (valid) {
            out.append(s, i, seq_len);
            i += seq_len;
        } else {
            out += "\xEF\xBF\xBD";
            i++;  // only skip lead byte; next byte may be a valid start
        }
    }
    return out;
}

}  // namespace dflash::common
