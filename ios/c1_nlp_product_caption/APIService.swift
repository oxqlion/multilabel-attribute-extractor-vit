// APIService.swift
// c1_nlp_product_caption
//
// Handles all HTTP communication with the FastAPI backend.
// Uses async/await + URLSession — no third-party dependencies.

import Foundation
import UIKit

// =============================================================================
// MARK: - APIService
// =============================================================================

final class APIService {

    static let shared = APIService()
    private init() {}

    // Shared URLSession with custom timeout
    private lazy var session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest  = Config.requestTimeoutSeconds
        cfg.timeoutIntervalForResource = Config.requestTimeoutSeconds + 10
        return URLSession(configuration: cfg)
    }()

    // =========================================================================
    // MARK: - POST /describe
    // =========================================================================

    /// Upload an image and return the full describe response.
    func describe(image: UIImage) async throws -> DescribeResponse {

        // ── Compress image to JPEG ────────────────────────────────────────
        guard let imageData = image.jpegData(
            compressionQuality: Config.imageCompressionQuality
        ) else {
            throw APIError.imageEncodingFailed
        }

        // ── Build multipart/form-data request ─────────────────────────────
        let boundary   = "Boundary-\(UUID().uuidString)"
        var request    = URLRequest(url: Config.describeURL)
        request.httpMethod  = "POST"
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        request.httpBody = buildMultipartBody(
            data:       imageData,
            fieldName:  "file",
            fileName:   "product.jpg",
            mimeType:   "image/jpeg",
            boundary:   boundary
        )

        // ── Send ──────────────────────────────────────────────────────────
        let (data, response) = try await session.data(for: request)

        // ── Validate HTTP status ──────────────────────────────────────────
        try validateHTTPResponse(response, data: data)

        // ── Decode ───────────────────────────────────────────────────────
        return try decodeJSON(DescribeResponse.self, from: data)
    }

    // =========================================================================
    // MARK: - GET /health
    // =========================================================================

    func health() async throws -> HealthResponse {
        let request = URLRequest(url: Config.healthURL)
        let (data, response) = try await session.data(for: request)
        try validateHTTPResponse(response, data: data)
        return try decodeJSON(HealthResponse.self, from: data)
    }

    // =========================================================================
    // MARK: - Helpers
    // =========================================================================

    /// Build a minimal multipart/form-data body with a single file field.
    private func buildMultipartBody(
        data:      Data,
        fieldName: String,
        fileName:  String,
        mimeType:  String,
        boundary:  String
    ) -> Data {
        var body = Data()
        let crlf = "\r\n"

        body.append("--\(boundary)\(crlf)")
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileName)\"\(crlf)")
        body.append("Content-Type: \(mimeType)\(crlf)")
        body.append(crlf)
        body.append(data)
        body.append(crlf)
        body.append("--\(boundary)--\(crlf)")

        return body
    }

    private func validateHTTPResponse(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw APIError.serverUnreachable("No HTTP response received.")
        }
        guard (200...299).contains(http.statusCode) else {
            // Try to extract a detail message from FastAPI's JSON error body
            let detail = (try? JSONDecoder().decode(FastAPIError.self, from: data))?.detail
                ?? String(data: data, encoding: .utf8)
                ?? "Unknown error"
            throw APIError.httpError(http.statusCode, detail)
        }
    }

    private func decodeJSON<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(type, from: data)
        } catch {
            let raw = String(data: data, encoding: .utf8) ?? "(binary)"
            throw APIError.decodingFailed("\(error)\n\nRaw response:\n\(raw.prefix(300))")
        }
    }
}

// FastAPI returns {"detail": "..."} for error responses
private struct FastAPIError: Decodable {
    let detail: String
}

// ── Data extension for appending strings ─────────────────────────────────────
private extension Data {
    mutating func append(_ string: String) {
        if let d = string.data(using: .utf8) {
            append(d)
        }
    }
}
