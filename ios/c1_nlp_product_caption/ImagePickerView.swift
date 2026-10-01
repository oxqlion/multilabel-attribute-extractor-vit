// ImagePickerView.swift
// c1_nlp_product_caption
//
// UIViewControllerRepresentable wrapper around UIImagePickerController.
// Supports both .photoLibrary and .camera source types.

import SwiftUI
import UIKit

struct ImagePickerView: UIViewControllerRepresentable {

    let sourceType: UIImagePickerController.SourceType
    let onImagePicked: (UIImage) -> Void

    // ── Coordinator (delegate) ────────────────────────────────────────────

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePickerView

        init(_ parent: ImagePickerView) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            // Prefer edited image if available (e.g. after camera crop)
            if let image = info[.editedImage] as? UIImage
                ?? info[.originalImage] as? UIImage {
                parent.onImagePicked(image)
            }
            picker.dismiss(animated: true)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    // ── UIViewControllerRepresentable ─────────────────────────────────────

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker            = UIImagePickerController()
        picker.sourceType     = sourceType
        picker.delegate       = context.coordinator
        picker.allowsEditing  = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // No dynamic updates needed
    }
}
