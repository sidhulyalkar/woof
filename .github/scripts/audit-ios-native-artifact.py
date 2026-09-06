#!/usr/bin/env python3
import json
import plistlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOBILE = ROOT / "apps" / "mobile"
IOS = MOBILE / "ios"
REPORT = ROOT / "ios-native-artifact-report.json"

EXPECTED_PERMISSION_STRINGS = {
    "NSCameraUsageDescription": "Woof uses the camera only when you choose to take a pet photo.",
    "NSPhotoLibraryUsageDescription": "Woof uses your photo library only when you choose a pet or profile photo.",
    "NSLocationWhenInUseUsageDescription": "Woof can use your location while the app is open to find nearby pet-friendly places and matches.",
}
EXPECTED_BUNDLE_ID = "com.woof.app"


def load_plist(path: Path):
    with path.open("rb") as handle:
        return plistlib.load(handle)


def privacy_manifest_summary(path: Path):
    data = load_plist(path)
    accessed = []
    for row in data.get("NSPrivacyAccessedAPITypes", []) or []:
        if not isinstance(row, dict):
            continue
        accessed.append(
            {
                "type": row.get("NSPrivacyAccessedAPIType"),
                "reasons": sorted(row.get("NSPrivacyAccessedAPITypeReasons", []) or []),
            }
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "tracking": data.get("NSPrivacyTracking"),
        "trackingDomains": data.get("NSPrivacyTrackingDomains", []) or [],
        "accessedAPITypes": sorted(accessed, key=lambda row: str(row.get("type"))),
        "collectedDataTypeCount": len(data.get("NSPrivacyCollectedDataTypes", []) or []),
    }


if not IOS.is_dir():
    raise SystemExit("generated iOS directory is missing; expo prebuild did not materialize native authority")

info_candidates = sorted(IOS.glob("*/Info.plist"))
if len(info_candidates) != 1:
    raise SystemExit(f"expected exactly one generated app Info.plist, found {len(info_candidates)}")
info_path = info_candidates[0]
info = load_plist(info_path)

for key, expected in EXPECTED_PERMISSION_STRINGS.items():
    actual = info.get(key)
    if actual != expected:
        raise SystemExit(f"generated Info.plist {key} mismatch: expected {expected!r}, got {actual!r}")

project_files = sorted(IOS.glob("*.xcodeproj/project.pbxproj"))
if len(project_files) != 1:
    raise SystemExit(f"expected exactly one generated Xcode project, found {len(project_files)}")
project_path = project_files[0]
project_text = project_path.read_text(encoding="utf-8")
bundle_matches = sorted(set(re.findall(r"PRODUCT_BUNDLE_IDENTIFIER = ([^;]+);", project_text)))
if EXPECTED_BUNDLE_ID not in bundle_matches:
    raise SystemExit(
        f"generated Xcode project does not contain bundle id {EXPECTED_BUNDLE_ID!r}; found {bundle_matches!r}"
    )

app_privacy_candidates = sorted(IOS.glob("*/PrivacyInfo.xcprivacy"))
app_privacy = [privacy_manifest_summary(path) for path in app_privacy_candidates]

resolved_dependency_manifests = {}
search_roots = [ROOT / "node_modules" / ".pnpm", MOBILE / "node_modules"]
for search_root in search_roots:
    if not search_root.exists():
        continue
    for path in search_root.rglob("PrivacyInfo.xcprivacy"):
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if IOS in resolved.parents:
            continue
        resolved_dependency_manifests[str(resolved)] = path

dependency_privacy = []
parse_errors = []
for resolved, display_path in sorted(resolved_dependency_manifests.items()):
    try:
        dependency_privacy.append(privacy_manifest_summary(Path(resolved)))
    except Exception as exc:  # pragma: no cover - evidence capture for third-party manifests
        parse_errors.append({"path": str(display_path), "error": str(exc)})

required_reason_union = {}
for manifest in dependency_privacy:
    for row in manifest["accessedAPITypes"]:
        api_type = row.get("type")
        if not api_type:
            continue
        required_reason_union.setdefault(api_type, set()).update(row.get("reasons") or [])

report = {
    "generatedInfoPlist": str(info_path.relative_to(ROOT)),
    "generatedXcodeProject": str(project_path.relative_to(ROOT)),
    "bundleIdentifiers": bundle_matches,
    "permissionDescriptions": {key: info.get(key) for key in EXPECTED_PERMISSION_STRINGS},
    "appPrivacyManifests": app_privacy,
    "dependencyPrivacyManifestCount": len(dependency_privacy),
    "dependencyPrivacyManifests": dependency_privacy,
    "dependencyPrivacyParseErrors": parse_errors,
    "dependencyRequiredReasonUnion": {
        key: sorted(values) for key, values in sorted(required_reason_union.items())
    },
}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"generated Info.plist: {report['generatedInfoPlist']}")
print(f"generated Xcode project: {report['generatedXcodeProject']}")
print(f"app privacy manifests: {len(app_privacy)}")
print(f"dependency privacy manifests: {len(dependency_privacy)}")
print(json.dumps(report["dependencyRequiredReasonUnion"], indent=2, sort_keys=True))

if parse_errors:
    raise SystemExit(f"could not parse {len(parse_errors)} dependency privacy manifest(s); inspect report")
if not app_privacy:
    raise SystemExit(
        "generated app target has no PrivacyInfo.xcprivacy; inspect dependency privacy inventory before declaring required reasons"
    )

print("iOS generated native artifact audit: OK")
