#!/usr/bin/env python3
import json
import plistlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOBILE = ROOT / "apps" / "mobile"
IOS = MOBILE / "ios"
REPORT = ROOT / "ios-cocoapods-build-report.json"
LOCK = IOS / "Podfile.lock"
PODS_PROJECT = IOS / "Pods" / "Pods.xcodeproj" / "project.pbxproj"
EXPECTED_BUNDLE_ID = "com.woof.app"


def load_plist(path: Path):
    with path.open("rb") as handle:
        return plistlib.load(handle)


def reason_union_from_rows(rows):
    union = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        api_type = row.get("NSPrivacyAccessedAPIType")
        if not api_type:
            continue
        union.setdefault(api_type, set()).update(row.get("NSPrivacyAccessedAPITypeReasons", []) or [])
    return {key: sorted(values) for key, values in sorted(union.items())}


def display(path: Path | None):
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


errors = []
source = json.loads((MOBILE / "app.json").read_text(encoding="utf-8"))
source_privacy = source.get("expo", {}).get("ios", {}).get("privacyManifests", {}) or {}
expected_reasons = reason_union_from_rows(source_privacy.get("NSPrivacyAccessedAPITypes", []))

lock_text = ""
if not LOCK.is_file():
    errors.append("Podfile.lock was not generated")
else:
    lock_text = LOCK.read_text(encoding="utf-8")

for forbidden in ["GoogleMaps", "Google-Maps-iOS-Utils", "react-native-maps/Google"]:
    if forbidden in lock_text:
        errors.append(f"unexpected Google Maps pod authority resolved: {forbidden}")

resolved_pods = sorted(
    {
        match.group(1)
        for match in re.finditer(r"^  - ([A-Za-z0-9_.+/-]+)(?: \(|:)", lock_text, re.MULTILINE)
    }
)

pods_project_text = PODS_PROJECT.read_text(encoding="utf-8") if PODS_PROJECT.is_file() else ""
privacy_resource_references = sorted(
    set(re.findall(r"[^\n]*PrivacyInfo\.xcprivacy[^\n]*", pods_project_text))
)

built_apps = sorted(
    path
    for path in (IOS / "build" / "DerivedData" / "Build" / "Products").glob("**/Woof.app")
    if path.is_dir() and "iphonesimulator" in str(path.parent)
)
if len(built_apps) != 1:
    errors.append(f"expected exactly one built iOS Simulator Woof.app, found {len(built_apps)}")
    built_app = built_apps[0] if built_apps else None
else:
    built_app = built_apps[0]

built_info = {}
built_privacy = {}
actual_reasons = {}
if built_app:
    info_path = built_app / "Info.plist"
    privacy_path = built_app / "PrivacyInfo.xcprivacy"
    if not info_path.is_file():
        errors.append("built app is missing root Info.plist")
    else:
        built_info = load_plist(info_path)
        if built_info.get("CFBundleIdentifier") != EXPECTED_BUNDLE_ID:
            errors.append(
                f"built app bundle id mismatch: expected {EXPECTED_BUNDLE_ID!r}, "
                f"got {built_info.get('CFBundleIdentifier')!r}"
            )
    if not privacy_path.is_file():
        errors.append("built app is missing root PrivacyInfo.xcprivacy")
    else:
        built_privacy = load_plist(privacy_path)
        actual_reasons = reason_union_from_rows(built_privacy.get("NSPrivacyAccessedAPITypes", []))
        if actual_reasons != expected_reasons:
            errors.append(
                "built app required-reason manifest differs from app-config authority: "
                f"expected={expected_reasons!r}, actual={actual_reasons!r}"
            )

report = {
    "podfileLock": display(LOCK if LOCK.is_file() else None),
    "resolvedPodCount": len(resolved_pods),
    "resolvedPods": resolved_pods,
    "googleMapsResolved": any(
        marker in lock_text for marker in ["GoogleMaps", "Google-Maps-iOS-Utils", "react-native-maps/Google"]
    ),
    "podsPrivacyResourceReferenceCount": len(privacy_resource_references),
    "podsPrivacyResourceReferences": privacy_resource_references,
    "builtApp": display(built_app),
    "builtBundleIdentifier": built_info.get("CFBundleIdentifier"),
    "expectedRequiredReasonUnion": expected_reasons,
    "builtRequiredReasonUnion": actual_reasons,
    "builtPrivacyTracking": built_privacy.get("NSPrivacyTracking") if built_privacy else None,
    "builtPrivacyTrackingDomains": built_privacy.get("NSPrivacyTrackingDomains", []) if built_privacy else [],
    "errors": errors,
}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"resolved pods: {len(resolved_pods)}")
print(f"Pods privacy resource references: {len(privacy_resource_references)}")
print(f"built app: {report['builtApp']}")
print(json.dumps(actual_reasons, indent=2, sort_keys=True))

if errors:
    print("CocoaPods/Xcode qualification discrepancies:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("iOS CocoaPods + built app authority: OK")
