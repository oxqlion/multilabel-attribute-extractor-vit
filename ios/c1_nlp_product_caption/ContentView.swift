// ContentView.swift
// c1_nlp_product_caption
//
// Main view — image picker → generate → show attributes + description

import SwiftUI
import Combine

struct ContentView: View {

    @StateObject private var viewModel = CaptionViewModel()
    @State private var showImageSourceSheet = false
    @State private var imageSource: UIImagePickerController.SourceType = .photoLibrary
    @State private var showImagePicker = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {

                    // ── Header ────────────────────────────────────────────
                    headerView

                    // ── Image picker card ─────────────────────────────────
                    imageCard

                    // ── Generate button ───────────────────────────────────
                    generateButton

                    // ── Results ───────────────────────────────────────────
                    if viewModel.state != .idle {
                        resultsSection
                    }

                    Spacer(minLength: 40)
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
            }
            .background(Color(.systemGroupedBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    navTitle
                }
            }
            // ── Image source action sheet ──────────────────────────────
            .confirmationDialog(
                "Choose Image Source",
                isPresented: $showImageSourceSheet,
                titleVisibility: .visible
            ) {
                Button("Photo Library") {
                    imageSource  = .photoLibrary
                    showImagePicker = true
                }
                if UIImagePickerController.isSourceTypeAvailable(.camera) {
                    Button("Camera") {
                        imageSource  = .camera
                        showImagePicker = true
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
            // ── Image picker sheet ─────────────────────────────────────
            .sheet(isPresented: $showImagePicker) {
                ImagePickerView(sourceType: imageSource) { image in
                    viewModel.selectImage(image)
                }
                .ignoresSafeArea()
            }
        }
    }

    // =========================================================
    // MARK: - Sub-views
    // =========================================================

    private var navTitle: some View {
        HStack(spacing: 6) {
            Image(systemName: "tshirt.fill")
                .foregroundStyle(.indigo)
            Text("FashionAI")
                .font(.headline)
                .fontWeight(.semibold)
        }
    }

    private var headerView: some View {
        VStack(spacing: 6) {
            Text("Product Caption Generator")
                .font(.title2)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
            Text("Upload a fashion photo to extract attributes\nand generate a product description.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.top, 8)
    }

    // ── Image card ────────────────────────────────────────────────────────
    private var imageCard: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
                .shadow(color: .black.opacity(0.06), radius: 8, y: 3)

            if let image = viewModel.selectedImage {
                // Selected image
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(maxWidth: .infinity)
                    .frame(height: 300)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .overlay(alignment: .bottomTrailing) {
                        Button {
                            showImageSourceSheet = true
                        } label: {
                            Label("Change", systemImage: "arrow.triangle.2.circlepath")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                        .padding(12)
                    }
            } else {
                // Empty state placeholder
                Button {
                    showImageSourceSheet = true
                } label: {
                    VStack(spacing: 14) {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 52))
                            .foregroundStyle(.indigo.opacity(0.8))
                        Text("Tap to select a fashion photo")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text("JPG, PNG, HEIC")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 220)
                }
                .buttonStyle(.plain)
            }
        }
    }

    // ── Generate button ───────────────────────────────────────────────────
    @ViewBuilder
    private var generateButton: some View {
        Button {
            Task { await viewModel.generate() }
        } label: {
            HStack(spacing: 10) {
                if viewModel.state == .loading {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.9)
                } else {
                    Image(systemName: "sparkles")
                }
                Text(viewModel.state == .loading ? "Analysing…" : "Generate Caption")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 52)
            .background(
                viewModel.canGenerate
                    ? LinearGradient(
                        colors: [.indigo, .purple],
                        startPoint: .leading, endPoint: .trailing)
                    : LinearGradient(
                        colors: [.gray.opacity(0.4), .gray.opacity(0.4)],
                        startPoint: .leading, endPoint: .trailing)
            )
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .shadow(
                color: viewModel.canGenerate ? .indigo.opacity(0.35) : .clear,
                radius: 8, y: 4)
        }
        .disabled(!viewModel.canGenerate)
        .animation(.easeInOut(duration: 0.2), value: viewModel.canGenerate)
    }

    // ── Results section ───────────────────────────────────────────────────
    @ViewBuilder
    private var resultsSection: some View {
        switch viewModel.state {

        case .loading:
            LoadingCard()

        case .error(let message):
            ErrorCard(message: message) {
                Task { await viewModel.generate() }
            }

        case .success(let result):
            VStack(spacing: 16) {
                DescriptionCard(description: result.description)
                AttributesCard(attributesByGroup: result.attributesByGroup)
                MetaCard(meta: result.meta)
            }

        case .idle:
            EmptyView()
        }
    }
}

// =============================================================================
// MARK: - Loading Card
// =============================================================================

struct LoadingCard: View {
    @State private var dots = ""
    let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        CardContainer {
            VStack(spacing: 12) {
                ProgressView()
                    .scaleEffect(1.3)
                    .tint(.indigo)
                Text("Extracting attributes\(dots)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
        }
        .onReceive(timer) { _ in
            dots = dots.count >= 3 ? "" : dots + "."
        }
    }
}

// =============================================================================
// MARK: - Error Card
// =============================================================================

struct ErrorCard: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        CardContainer {
            VStack(spacing: 14) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title)
                    .foregroundStyle(.orange)

                Text("Something went wrong")
                    .font(.headline)

                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                Button("Try Again", action: onRetry)
                    .buttonStyle(.borderedProminent)
                    .tint(.indigo)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
    }
}

// =============================================================================
// MARK: - Description Card
// =============================================================================

struct DescriptionCard: View {
    let description: String
    @State private var copied = false

    var body: some View {
        CardContainer {
            VStack(alignment: .leading, spacing: 14) {

                // Header row
                HStack {
                    Label("Product Description", systemImage: "text.alignleft")
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Spacer()
                    Button {
                        UIPasteboard.general.string = description
                        withAnimation { copied = true }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
                            withAnimation { copied = false }
                        }
                    } label: {
                        Label(
                            copied ? "Copied!" : "Copy",
                            systemImage: copied ? "checkmark" : "doc.on.doc"
                        )
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(copied ? .green : .indigo)
                    }
                    .animation(.spring(duration: 0.3), value: copied)
                }

                Divider()

                Text(description)
                    .font(.body)
                    .lineSpacing(5)
                    .foregroundStyle(.primary)
            }
        }
    }
}

// =============================================================================
// MARK: - Attributes Card
// =============================================================================

struct AttributesCard: View {
    let attributesByGroup: [String: [DetectedAttribute]]
    @State private var expandedGroups: Set<String> = []

    // Sorted groups — put most informative ones first
    private var sortedGroups: [(String, [DetectedAttribute])] {
        attributesByGroup
            .sorted { lhs, rhs in lhs.value.first?.confidence ?? 0 > rhs.value.first?.confidence ?? 0 }
    }

    var body: some View {
        CardContainer {
            VStack(alignment: .leading, spacing: 14) {

                // Header
                HStack {
                    Label("Detected Attributes", systemImage: "tag.fill")
                        .font(.headline)
                    Spacer()
                    Text("\(attributesByGroup.values.flatMap { $0 }.count) total")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Divider()

                // Attribute groups
                VStack(spacing: 8) {
                    ForEach(sortedGroups, id: \.0) { group, attrs in
                        AttributeGroupRow(
                            groupName:  group,
                            attributes: attrs,
                            isExpanded: expandedGroups.contains(group)
                        ) {
                            withAnimation(.spring(duration: 0.3)) {
                                if expandedGroups.contains(group) {
                                    expandedGroups.remove(group)
                                } else {
                                    expandedGroups.insert(group)
                                }
                            }
                        }
                    }
                }
            }
        }
        .onAppear {
            // Auto-expand top 3 groups
            let top = sortedGroups.prefix(3).map { $0.0 }
            expandedGroups = Set(top)
        }
    }
}

struct AttributeGroupRow: View {
    let groupName:  String
    let attributes: [DetectedAttribute]
    let isExpanded: Bool
    let onTap:      () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Group header (tappable)
            Button(action: onTap) {
                HStack {
                    Circle()
                        .fill(groupColor(groupName))
                        .frame(width: 9, height: 9)
                    Text(groupName.capitalized)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(.primary)
                    Spacer()
                    Text("\(attributes.count)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color(.tertiarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(.plain)

            // Expanded attribute chips
            if isExpanded {
                FlowLayout(spacing: 8) {
                    ForEach(attributes) { attr in
                        AttributeChip(attribute: attr)
                    }
                }
                .padding(.top, 8)
                .padding(.horizontal, 4)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    private func groupColor(_ group: String) -> Color {
        // Deterministic colour per supercategory
        let palette: [Color] = [.indigo, .purple, .teal, .orange, .pink, .green, .blue, .red]
        let idx = abs(group.hashValue) % palette.count
        return palette[idx]
    }
}

struct AttributeChip: View {
    let attribute: DetectedAttribute

    // Confidence → colour intensity
    private var confidenceColor: Color {
        switch attribute.confidence {
        case 0.8...: return .green
        case 0.6..<0.8: return .indigo
        case 0.4..<0.6: return .orange
        default: return .gray
        }
    }

    var body: some View {
        HStack(spacing: 5) {
            Text(attribute.name)
                .font(.caption)
                .fontWeight(.medium)
            Text(String(format: "%.0f%%", attribute.confidence * 100))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(confidenceColor.opacity(0.12))
        .foregroundStyle(confidenceColor)
        .clipShape(Capsule())
        .overlay(Capsule().stroke(confidenceColor.opacity(0.3), lineWidth: 1))
    }
}

// =============================================================================
// MARK: - Meta Card (timing / model info, collapsed by default)
// =============================================================================

struct MetaCard: View {
    let meta: InferenceMeta
    @State private var isExpanded = false

    var body: some View {
        CardContainer {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    withAnimation(.spring(duration: 0.3)) {
                        isExpanded.toggle()
                    }
                } label: {
                    HStack {
                        Label("Model Info", systemImage: "info.circle")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.2fs", meta.timing.totalS))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)

                if isExpanded {
                    Divider()
                    VStack(spacing: 8) {
                        MetaRow(label: "Image size",     value: meta.imageSize)
                        MetaRow(label: "Attributes raw", value: "\(meta.nAttributesRaw)")
                        MetaRow(label: "Threshold",      value: String(format: "%.2f", meta.thresholdUsed))
                        MetaRow(label: "LLM",            value: meta.llmModel)
                        MetaRow(label: "Detection",      value: String(format: "%.0fms", meta.timing.detectionS * 1000))
                        MetaRow(label: "Attribute",      value: String(format: "%.0fms", meta.timing.attributeS * 1000))
                        MetaRow(label: "LLM",            value: String(format: "%.2fs",  meta.timing.llmS))
                        MetaRow(label: "Total",          value: String(format: "%.2fs",  meta.timing.totalS))
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
        }
    }
}

struct MetaRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundStyle(.primary)
        }
    }
}

// =============================================================================
// MARK: - Shared card container
// =============================================================================

struct CardContainer<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        content
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: .black.opacity(0.05), radius: 6, y: 2)
    }
}

// =============================================================================
// MARK: - Flow layout (wrapping chip layout)
// =============================================================================

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        let height = rows.map { row in
            row.map { subviews[$0].sizeThatFits(.unspecified).height }.max() ?? 0
        }.reduce(0) { $0 + $1 + spacing } - spacing
        return CGSize(width: proposal.width ?? 0, height: max(height, 0))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX
            let rowHeight = row.map { subviews[$0].sizeThatFits(.unspecified).height }.max() ?? 0
            for idx in row {
                let size = subviews[idx].sizeThatFits(.unspecified)
                subviews[idx].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += rowHeight + spacing
        }
    }

    private func computeRows(proposal: ProposedViewSize, subviews: Subviews) -> [[Int]] {
        var rows: [[Int]] = [[]]
        var x: CGFloat = 0
        let maxW = proposal.width ?? .infinity
        for (i, subview) in subviews.enumerated() {
            let w = subview.sizeThatFits(.unspecified).width
            if x + w > maxW, !rows[rows.count - 1].isEmpty {
                rows.append([])
                x = 0
            }
            rows[rows.count - 1].append(i)
            x += w + spacing
        }
        return rows.filter { !$0.isEmpty }
    }
}

#Preview {
    ContentView()
}
