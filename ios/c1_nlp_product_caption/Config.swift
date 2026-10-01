// Config.swift
// c1_nlp_product_caption
//
// ⚠️  IMPORTANT — READ THIS BEFORE RUNNING ON A PHYSICAL IPHONE
// ──────────────────────────────────────────────────────────────
// Your iPhone cannot reach "localhost" — localhost means the iPhone itself.
// To connect to the FastAPI server running on your Mac, you need to use
// your Mac's local Wi-Fi IP address instead.
//
// HOW TO FIND YOUR MAC'S IP ADDRESS:
//   1. Open System Settings → Network → Wi-Fi → Details…
//   2. Look for the IPv4 Address (e.g. 192.168.1.42)
//   OR open Terminal and run:
//       ipconfig getifaddr en0
//
// Both your Mac and iPhone MUST be on the same Wi-Fi network.
//
// Then update API_BASE_URL below with that IP address.
// Example:  "http://192.168.1.42:8000"

import Foundation

enum Config {

    // ── Change this to your Mac's local Wi-Fi IP ──────────────────────────
    static let API_BASE_URL = "https://g0b6vs5v-8000.asse.devtunnels.ms"
    // ─────────────────────────────────────────────────────────────────────

    // Endpoints
    static var describeURL: URL {
        URL(string: "\(API_BASE_URL)/describe")!
    }
    static var healthURL: URL {
        URL(string: "\(API_BASE_URL)/health")!
    }

    // Image compression quality sent to the API
    // 0.85 gives good quality at ~200–600KB for most product photos
    static let imageCompressionQuality: CGFloat = 0.85

    // Request timeout in seconds
    // LLM generation takes ~1.5–2s on M5, so 30s is a safe margin
    static let requestTimeoutSeconds: TimeInterval = 30
}
