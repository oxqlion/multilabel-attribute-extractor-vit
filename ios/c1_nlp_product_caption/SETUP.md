# iOS App — Setup Guide

## Project Structure (what to copy into Xcode)

```
c1_nlp_product_caption/          ← your existing Xcode project root
  c1_nlp_product_caption/
    c1_nlp_product_captionApp.swift   ← REPLACE existing file
    ContentView.swift                  ← REPLACE existing file
    CaptionViewModel.swift             ← ADD new file
    APIService.swift                   ← ADD new file
    Models.swift                       ← ADD new file
    ImagePickerView.swift              ← ADD new file
    Config.swift                       ← ADD new file
    Assets.xcassets/                   ← leave as-is
```

---

## Step-by-step Xcode Setup

### 1. Replace ContentView.swift and App file
- In Xcode, click `ContentView.swift` → select all → paste new code
- Click `c1_nlp_product_captionApp.swift` → select all → paste new code

### 2. Add the new Swift files
For each new file (CaptionViewModel, APIService, Models, ImagePickerView, Config):

1. In Xcode: **File → New → File from Template → Swift File**
2. Name it exactly (e.g. `CaptionViewModel.swift`)
3. Make sure the target `c1_nlp_product_caption` is checked
4. Paste the code

### 3. Add required Info.plist keys (CRITICAL for camera + photo library)

In Xcode:
1. Click your project name (blue icon) in the Navigator
2. Select the **c1_nlp_product_caption** target
3. Go to **Info** tab
4. Click **+** to add these two keys:

| Key | Type | Value |
|---|---|---|
| `Privacy - Photo Library Usage Description` | String | `FashionAI needs access to your photo library to select product images.` |
| `Privacy - Camera Usage Description` | String | `FashionAI uses the camera to capture product photos for analysis.` |

⚠️ Without these, the app will crash when the user taps the image picker.

### 4. Allow HTTP connections to local server (App Transport Security)

By default, iOS blocks HTTP (non-HTTPS) connections. Your local backend runs
on HTTP, so you need to add an exception.

In Xcode:
1. Open `Info.plist` (right-click → Open As → Source Code)
2. Add this inside the root `<dict>`:

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
    <key>NSExceptionDomains</key>
    <dict>
        <key>192.168.1.100</key>
        <dict>
            <key>NSExceptionAllowsInsecureHTTPLoads</key>
            <true/>
            <key>NSIncludesSubdomains</key>
            <false/>
        </dict>
    </dict>
</dict>
```

Replace `192.168.1.100` with your Mac's actual IP address.

### 5. Set your Mac's IP in Config.swift

```swift
// Config.swift
static let API_BASE_URL = "http://192.168.1.YOUR_IP:8000"
```

**Find your Mac's IP:**
```bash
ipconfig getifaddr en0
```

---

## Start the backend before running the app

```bash
# In your fashionpedia_api directory:
cd fashionpedia_api
source .venv/bin/activate

# IMPORTANT: use --host 0.0.0.0 so your iPhone can reach it
uvicorn main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` means "listen on all network interfaces" — without this,
the server only accepts connections from localhost (your Mac itself), and
your iPhone will be refused.

---

## Run on physical iPhone

1. Connect iPhone via USB (or use wireless debugging)
2. In Xcode: select your iPhone as the run destination (top bar)
3. **Product → Run** (⌘R)
4. First launch: on iPhone go to **Settings → General → VPN & Device Management**
   → trust your developer certificate

---

## Quick connectivity test (before opening the app)

On your iPhone, open Safari and visit:
```
http://192.168.1.YOUR_IP:8000/health
```
You should see:
```json
{"status": "ok", "models_loaded": true, ...}
```

If you get an error, the connection isn't working yet — check Wi-Fi and
the `--host 0.0.0.0` flag.

---

## Minimum Requirements

| Requirement | Value |
|---|---|
| iOS | 16.0+ (SwiftUI Layout protocol) |
| Xcode | 15.0+ |
| Swift | 5.9+ |
| Backend | FastAPI running on same Wi-Fi |
