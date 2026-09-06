#!/usr/bin/env python3
import json
import plistlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOBILE = ROOT / "apps" / "mobile"
IOS = MOBILE / "ios"
REPORT = ROOT / "ios-native-artifact-report.json"
APP_CONFIG = MOBILE / "app.json"
EXPECTED_BUNDLE_ID = "com.woof.app"

PERMISSION_PLUGIN_AUTHORITY = {
    "NSCameraUsageDescription": ("expo-camera", "cameraPermission"),
    "NSPhotoLibraryUsageDescription": ("expo-image-picker", "photosPermission"),
    "NSLocationWhenInUseUsageDescription": ("expo-location", "locationWhenInUsePermission"),
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
        "path": display_path(path),
        "tracking": data.get("NSPrivacyTracking"),
        "trackingDomains": data.get("NSPrivacyTrackingDomains", []) or [],
        "accessedAPITypes": sorted(accessed, key=lambda row: str(row.get("type"))),
        "collectedDataTypeCount": len(data.get("NSPrivacyCollectedDataTypes", []) or []),
    }


def plugin_options(expo_config: dict) -> dict[str, dict]:
    resolved = {}
    for plugin in expo_config.get("plugins", []) or []:
        if not isinstance(plugin, list) or len(plugin) < 2:
            continue
        name, options = plugin[0], plugin[1]
        if isinstance(name, str) and isinstance(options, dict):
            resolved[name] = options
    return resolved


source = json.loads(APP_CONFIG.read_text(encoding="utf-8"))
expo = source.get("expo", {})
source_info = expo.get("ios", {}).get("infoPlist", {}) or {}
plugins = plugin_options(expo)
expected_permissions = {
    key: source_info.get(key) for key in PERMISSION_PLUGIN_AUTHORITY
}
source_permission_conflicts = []
errors = []

for key, (plugin_name, option_name) in PERMISSION_PLUGIN_AUTHORITY.items():
    expected = expected_permissions.get(key)
    plugin_value = plugins.get(plugin_name, {}).get(option_name)
    if not isinstance(expected, str) or not expected.strip():
        source_permission_conflicts.append(
            f"app.json ios.infoPlist is missing non-empty {key}"
        )
    if expected != plugin_value:
        source_permission_conflicts.append(
            f"app.json permission authority conflict for {key}: "
            f"ios.infoPlist={expected!r}, {plugin_name}.{option_name}={plugin_value!r}"
        )

errors.extend(source_permission_conflicts)

info_path = None
info = {}
info_candidates = sorted(IOS.glob("*/Info.plist")) if IOS.is_dir() else []
if len(info_candidates) != 1:
    errors.append(f"expected exactly one generated app Info.plist, found {len(info_candidates)}")
else:
    info_path = info_candidates[0]
    try:
        info = load_plist(info_path)
    except Exception as exc:
        errors.append(f"could not parse generated Info.plist: {exc}")

for key, expected in expected_permissions.items():
    actual = info.get(key)
    if actual != expected:
        errors.append(
            f"generated Info.plist {key} mismatch: expected {expected!r}, got {actual!r}"
        )

project_path = None
project_text = ""
project_files = sorted(IOS.glob("*.xcodeproj/project.pbxproj")) if IOS.is_dir() else []
if len(project_files) != 1:
    errors.append(f"expected exactly one generated Xcode project, found {len(project_files)}")
else:
    project_path = project_files[0]
    try:
        project_text = project_path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"could not read generated Xcode project: {exc}")

bundle_matches = sorted(set(re.findall(r"PRODUCT_BUNDLE_IDENTIFIER = ([^;]+);", project_text)))
if EXPECTED_BUNDLE_ID not in bundle_matches:
    errors.append(
        f"generated Xcode project does not contain bundle id {EXPECTED_BUNDLE_ID!r}; "
        f"found {bundle_matches!r}"
    )

app_privacy = []
app_privacy_parse_errors = []
app_privacy_candidates = sorted(IOS.glob("*/PrivacyInfo.xcprivacy")) if IOS.is_dir() else []
for path in app_privacy_candidates:
    try:
        app_privacy.append(privacy_manifest_summary(path))
    except Exception as exc:
        app_privacy_parse_errors.append({"path": display_path(path), "error": str(exc)})

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
        resolved_dependency_manifests[str(resolved)] = resolved

dependency_privacy = []
dependency_parse_errors = []
for resolved in sorted(resolved_dependency_manifests.values(), key=str):
    try:
        dependency_privacy.append(privacy_manifest_summary(resolved))
    except Exception as exc:  # pragma: no cover - evidence capture for third-party manifests
        dependency_parse_errors.append({"path": display_path(resolved), "error": str(exc)})

required_reason_union = {}
for manifest in dependency_privacy:
    for row in manifest["accessedAPITypes"]:
        api_type = row.get("type")
        if not api_type:
            continue
        required_reason_union.setdefault(api_type, set()).update(row.get("reasons") or [])

required_reason_union = {
    key: sorted(values) for key, values in sorted(required_reason_union.items())
}

if app_privacy_parse_errors:
    errors.append(
        f"could not parse {len(app_privacy_parse_errors)} generated app privacy manifest(s)"
    )
if dependency_parse_errors:
    errors.append(
        f"could not parse {len(dependency_parse_errors)} dependency privacy manifest(s)"
    )
if required_reason_union and not app_privacy:
    errors.append(
        "installed dependencies declare required-reason APIs but generated app target has no "
        "PrivacyInfo.xcprivacy; inspect dependencyRequiredReasonUnion before configuring app-level aggregation"
    )

report = {
    "sourcePermissionDescriptions": expected_permissions,
    "sourcePermissionConflicts": source_permission_conflicts,
    "generatedInfoPlist": display_path(info_path) if info_path else None,
    "generatedXcodeProject": display_path(project_path) if project_path else None,
    "bundleIdentifiers": bundle_matches,
    "permissionDescriptions": {key: info.get(key) for key in PERMISSION_PLUGIN_AUTHORITY},
    "appPrivacyManifests": app_privacy,
    "appPrivacyParseErrors": app_privacy_parse_errors,
    "dependencyPrivacyManifestCount": len(dependency_privacy),
    "dependencyPrivacyManifests": dependency_privacy,
    "dependencyPrivacyParseErrors": dependency_parse_errors,
    "dependencyRequiredReasonUnion": required_reason_union,
    "errors": errors,
}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"generated Info.plist: {report['generatedInfoPlist']}")
print(f"generated Xcode project: {report['generatedXcodeProject']}")
print(f"app privacy manifests: {len(app_privacy)}")
print(f"dependency privacy manifests: {len(dependency_privacy)}")
print(json.dumps(required_reason_union, indent=2, sort_keys=True))

if errors:
    print("native artifact audit discrepancies:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("iOS generated native artifact audit: OK")
