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
EXPO_AUTOLINK = ROOT / "expo-autolinking-apple.json"
RN_AUTOLINK = ROOT / "react-native-autolinking-ios.json"
PODFILE = IOS / "Podfile"
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


def load_json(path: Path, label: str, errors: list[str]) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"could not read {label}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"{label} must resolve to a JSON object")
        return {}
    return data


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


def package_root_for(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "package.json").is_file():
            return directory
    return None


def reason_union(manifests: list[dict]) -> dict[str, list[str]]:
    union: dict[str, set[str]] = {}
    for manifest in manifests:
        for row in manifest.get("accessedAPITypes", []):
            api_type = row.get("type")
            if not api_type:
                continue
            union.setdefault(api_type, set()).update(row.get("reasons") or [])
    return {key: sorted(values) for key, values in sorted(union.items())}


errors: list[str] = []
source = load_json(APP_CONFIG, "apps/mobile/app.json", errors)
expo = source.get("expo", {}) if isinstance(source.get("expo", {}), dict) else {}
source_info = expo.get("ios", {}).get("infoPlist", {}) or {}
plugins = plugin_options(expo)
expected_permissions = {key: source_info.get(key) for key in PERMISSION_PLUGIN_AUTHORITY}
source_permission_conflicts = []

for key, (plugin_name, option_name) in PERMISSION_PLUGIN_AUTHORITY.items():
    expected = expected_permissions.get(key)
    plugin_value = plugins.get(plugin_name, {}).get(option_name)
    if not isinstance(expected, str) or not expected.strip():
        source_permission_conflicts.append(f"app.json ios.infoPlist is missing non-empty {key}")
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

bundle_matches = sorted(
    {
        match.strip().strip('"')
        for match in re.findall(r"PRODUCT_BUNDLE_IDENTIFIER = ([^;]+);", project_text)
    }
)
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

expo_autolink = load_json(EXPO_AUTOLINK, "Expo Apple autolinking evidence", errors)
rn_autolink = load_json(RN_AUTOLINK, "React Native iOS autolinking evidence", errors)
linked_roots: dict[str, set[str]] = {}

for module in expo_autolink.get("modules", []) or []:
    if not isinstance(module, dict):
        continue
    module_name = module.get("packageName") or "unknown-expo-module"
    roots = set()
    direct_root = package_root_for(module.get("path"))
    if direct_root:
        roots.add(direct_root)
    for pod in module.get("pods", []) or []:
        if not isinstance(pod, dict):
            continue
        root = package_root_for(pod.get("podspecDir"))
        if root:
            roots.add(root)
    for root in roots:
        linked_roots.setdefault(str(root), set()).add(str(module_name))

react_native_root = package_root_for(rn_autolink.get("reactNativePath"))
if react_native_root:
    linked_roots.setdefault(str(react_native_root), set()).add("react-native")

for dependency_name, dependency in (rn_autolink.get("dependencies", {}) or {}).items():
    if not isinstance(dependency, dict):
        continue
    ios_config = (dependency.get("platforms", {}) or {}).get("ios")
    if not isinstance(ios_config, dict):
        continue
    root = package_root_for(dependency.get("root")) or package_root_for(ios_config.get("podspecPath"))
    if root:
        linked_roots.setdefault(str(root), set()).add(str(dependency_name))

podfile_text = PODFILE.read_text(encoding="utf-8") if PODFILE.is_file() else ""
google_maps_enabled = "react-native-maps/Google" in podfile_text

resolved_linked_manifests: dict[str, Path] = {}
excluded_optional_manifests = []
for root_value in sorted(linked_roots):
    root = Path(root_value)
    if not root.exists():
        continue
    for path in root.rglob("PrivacyInfo.xcprivacy"):
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if IOS in resolved.parents:
            continue
        if (
            not google_maps_enabled
            and root.name == "react-native-maps"
            and "AirGoogleMaps" in resolved.parts
        ):
            excluded_optional_manifests.append(
                {
                    "path": display_path(resolved),
                    "reason": "react-native-maps Google subspec is not enabled in generated Podfile",
                }
            )
            continue
        resolved_linked_manifests[str(resolved)] = resolved

linked_privacy = []
linked_privacy_parse_errors = []
for resolved in sorted(resolved_linked_manifests.values(), key=str):
    try:
        linked_privacy.append(privacy_manifest_summary(resolved))
    except Exception as exc:  # pragma: no cover - evidence capture for third-party manifests
        linked_privacy_parse_errors.append({"path": display_path(resolved), "error": str(exc)})

linked_required_reason_union = reason_union(linked_privacy)
app_required_reason_union = reason_union(app_privacy)
missing_app_required_reasons = {}
for api_type, reasons in linked_required_reason_union.items():
    missing = sorted(set(reasons) - set(app_required_reason_union.get(api_type, [])))
    if missing:
        missing_app_required_reasons[api_type] = missing

if app_privacy_parse_errors:
    errors.append(
        f"could not parse {len(app_privacy_parse_errors)} generated app privacy manifest(s)"
    )
if linked_privacy_parse_errors:
    errors.append(
        f"could not parse {len(linked_privacy_parse_errors)} autolinked dependency privacy manifest(s)"
    )
if linked_required_reason_union and not app_privacy:
    errors.append(
        "autolinked native dependencies declare required-reason APIs but generated app target has no "
        "PrivacyInfo.xcprivacy; inspect autolinkedRequiredReasonUnion before configuring app-level aggregation"
    )
elif missing_app_required_reasons:
    errors.append(
        "generated app privacy manifest does not cover the autolinked dependency required-reason candidate set: "
        f"{missing_app_required_reasons!r}"
    )

report = {
    "sourcePermissionDescriptions": expected_permissions,
    "sourcePermissionConflicts": source_permission_conflicts,
    "generatedInfoPlist": display_path(info_path) if info_path else None,
    "generatedXcodeProject": display_path(project_path) if project_path else None,
    "generatedPodfile": display_path(PODFILE) if PODFILE.is_file() else None,
    "bundleIdentifiers": bundle_matches,
    "permissionDescriptions": {key: info.get(key) for key in PERMISSION_PLUGIN_AUTHORITY},
    "googleMapsSubspecEnabled": google_maps_enabled,
    "autolinkedPackageRoots": {
        display_path(Path(root)): sorted(names) for root, names in sorted(linked_roots.items())
    },
    "excludedOptionalPrivacyManifests": excluded_optional_manifests,
    "appPrivacyManifests": app_privacy,
    "appPrivacyParseErrors": app_privacy_parse_errors,
    "appRequiredReasonUnion": app_required_reason_union,
    "autolinkedPrivacyManifestCount": len(linked_privacy),
    "autolinkedPrivacyManifests": linked_privacy,
    "autolinkedPrivacyParseErrors": linked_privacy_parse_errors,
    "autolinkedRequiredReasonUnion": linked_required_reason_union,
    "missingAppRequiredReasons": missing_app_required_reasons,
    "errors": errors,
}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"generated Info.plist: {report['generatedInfoPlist']}")
print(f"generated Xcode project: {report['generatedXcodeProject']}")
print(f"app privacy manifests: {len(app_privacy)}")
print(f"autolinked privacy manifests: {len(linked_privacy)}")
print(f"react-native-maps Google subspec enabled: {google_maps_enabled}")
print(json.dumps(linked_required_reason_union, indent=2, sort_keys=True))

if errors:
    print("native artifact audit discrepancies:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("iOS generated native artifact audit: OK")
