// CaptionViewModel.swift
// c1_nlp_product_caption
//
// ObservableObject that drives the entire ContentView.
// Keeps UI state, selected image, and API result in one place.

import SwiftUI
import Combine

// =============================================================================
// MARK: - View state enum
// =============================================================================

enum ViewState: Equatable {
    case idle
    case loading
    case success(DescribeResponse)
    case error(String)

    static func == (lhs: ViewState, rhs: ViewState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.loading, .loading): return true
        case (.error(let a), .error(let b)):        return a == b
        case (.success, .success):                  return true  // simplified
        default:                                    return false
        }
    }
}

// =============================================================================
// MARK: - ViewModel
// =============================================================================

@MainActor
final class CaptionViewModel: ObservableObject {

    @Published var selectedImage: UIImage? = nil
    @Published var state: ViewState = .idle

    /// Button is enabled only when an image is selected and not currently loading
    var canGenerate: Bool {
        selectedImage != nil && state != .loading
    }

    // ── Image selection ───────────────────────────────────────────────────

    func selectImage(_ image: UIImage) {
        selectedImage = image
        // Reset results when a new image is picked
        state = .idle
    }

    // ── API call ──────────────────────────────────────────────────────────

    func generate() async {
        guard let image = selectedImage else {
            state = .error(APIError.noImageSelected.localizedDescription ?? "No image")
            return
        }

        state = .loading

        do {
            let result = try await APIService.shared.describe(image: image)
            state = .success(result)
        } catch let apiError as APIError {
            state = .error(apiError.localizedDescription ?? "Unknown error")
        } catch let urlError as URLError {
            // Map common URLErrors to friendly messages
            switch urlError.code {
            case .notConnectedToInternet:
                state = .error("No internet connection.")
            case .timedOut:
                state = .error("Request timed out. The server may be slow or unreachable.\n\nCheck that your Mac's backend is running.")
            case .cannotConnectToHost, .networkConnectionLost:
                state = .error(APIError.serverUnreachable(
                    "Could not connect to \(Config.API_BASE_URL).\n\nCheck that:\n• Mac and iPhone are on the same Wi-Fi\n• Backend is running: uvicorn main:app --host 0.0.0.0 --port 8000\n• Config.swift has the correct IP"
                ).localizedDescription ?? "Connection failed")
            default:
                state = .error("Network error: \(urlError.localizedDescription)")
            }
        } catch {
            state = .error("Unexpected error: \(error.localizedDescription)")
        }
    }
}
