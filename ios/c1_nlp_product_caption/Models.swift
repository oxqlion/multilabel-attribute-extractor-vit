// Models.swift
// c1_nlp_product_caption
//
// Swift data models that exactly mirror the FastAPI response schema.
// All property names use camelCase; JSON decoding uses snake_case keys.

import Foundation

// =============================================================================
// MARK: - API Response  (POST /describe)
// =============================================================================

struct DescribeResponse: Decodable {
    let attributes:         [DetectedAttribute]
    let attributesByGroup:  [String: [DetectedAttribute]]
    let description:        String
    let meta:               InferenceMeta

    enum CodingKeys: String, CodingKey {
        case attributes
        case attributesByGroup  = "attributes_by_group"
        case description
        case meta
    }
}

// =============================================================================
// MARK: - DetectedAttribute
// =============================================================================

struct DetectedAttribute: Decodable, Identifiable {
    let attrId:       Int
    let name:         String
    let supercategory: String
    let confidence:   Double

    // Identifiable conformance (needed for ForEach)
    var id: Int { attrId }

    enum CodingKeys: String, CodingKey {
        case attrId        = "attr_id"
        case name
        case supercategory
        case confidence
    }
}

// =============================================================================
// MARK: - InferenceMeta
// =============================================================================

struct InferenceMeta: Decodable {
    let imageSize:         String
    let cropBox:           CropBox
    let detectionStrategy: String
    let nAttributesRaw:    Int
    let thresholdUsed:     Double
    let attributeModel:    String
    let llmBackend:        String
    let llmModel:          String
    let timing:            Timing

    enum CodingKeys: String, CodingKey {
        case imageSize         = "image_size"
        case cropBox           = "crop_box"
        case detectionStrategy = "detection_strategy"
        case nAttributesRaw    = "n_attributes_raw"
        case thresholdUsed     = "threshold_used"
        case attributeModel    = "attribute_model"
        case llmBackend        = "llm_backend"
        case llmModel          = "llm_model"
        case timing
    }
}

struct CropBox: Decodable {
    let x:      Int
    let y:      Int
    let width:  Int
    let height: Int
}

struct Timing: Decodable {
    let detectionS: Double
    let attributeS: Double
    let llmS:       Double
    let totalS:     Double

    enum CodingKeys: String, CodingKey {
        case detectionS = "detection_s"
        case attributeS = "attribute_s"
        case llmS       = "llm_s"
        case totalS     = "total_s"
    }
}

// =============================================================================
// MARK: - Health response  (GET /health)
// =============================================================================

struct HealthResponse: Decodable {
    let status:        String
    let modelsLoaded:  Bool
    let llmBackend:    String
    let llmModel:      String
    let device:        String

    enum CodingKeys: String, CodingKey {
        case status
        case modelsLoaded  = "models_loaded"
        case llmBackend    = "llm_backend"
        case llmModel      = "llm_model"
        case device
    }
}

// =============================================================================
// MARK: - API errors
// =============================================================================

enum APIError: LocalizedError {
    case noImageSelected
    case imageEncodingFailed
    case serverUnreachable(String)
    case httpError(Int, String)
    case decodingFailed(String)
    case unknown(String)

    var errorDescription: String? {
        switch self {
        case .noImageSelected:
            return "No image selected. Please pick a photo first."
        case .imageEncodingFailed:
            return "Failed to compress image. Please try a different photo."
        case .serverUnreachable(let detail):
            return "Cannot reach the server.\n\n\(detail)\n\nMake sure:\n• Mac and iPhone are on the same Wi-Fi\n• Backend is running (uvicorn main:app)\n• API_BASE_URL is correct in Config.swift"
        case .httpError(let code, let message):
            return "Server error \(code): \(message)"
        case .decodingFailed(let detail):
            return "Unexpected response from server.\n\(detail)"
        case .unknown(let msg):
            return msg
        }
    }
}
