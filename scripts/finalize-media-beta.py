from pathlib import Path
import json


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor missing in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


def replace_all(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor missing in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new))


service = "apps/api/src/media-library/media-library.service.ts"
replace_once(
    service,
    "import { StorageService } from '../storage/storage.service';",
    "import { StorageService } from '../storage/storage.service';\nimport { declaredMimeMatches, sniffMediaType } from './media-sniff';",
)
replace_once(
    service,
    "  albumLinks?: Array<{ albumId: string }>;\n};",
    "  albumLinks?: Array<{ albumId: string }>;\n  derivatives?: Array<{ kind: string; storageKey: string; status: string }>;\n};",
)
replace_once(
    service,
    "    const sizeMatches = object.sizeBytes === Number(asset.sizeBytes);\n    const contentTypeMatches = !object.contentType || object.contentType === asset.mimeType;\n    if (!sizeMatches || !contentTypeMatches) {",
    "    const sizeMatches = object.sizeBytes === Number(asset.sizeBytes);\n    const contentTypeMatches = !object.contentType || object.contentType === asset.mimeType;\n    const signatureMatches = await this.storage\n      .getObjectHeader(asset.storageKey, 64)\n      .then((header) => declaredMimeMatches(sniffMediaType(header), asset.mimeType))\n      .catch(() => false);\n    if (!sizeMatches || !contentTypeMatches || !signatureMatches) {",
)
replace_once(
    service,
    "      throw new BadRequestException('Uploaded media did not match the declared size or content type');",
    "      throw new BadRequestException(\n        'Uploaded media did not match the declared size, type, or file signature',\n      );",
)

# Any asset read for library/detail receives only READY derivative object references.
derivative_include = """include: {
        albumLinks: { select: { albumId: true } },
        derivatives: {
          where: { status: 'READY' },
          select: { kind: true, storageKey: true, status: true },
        },
      },"""
replace_all(
    service,
    "include: { albumLinks: { select: { albumId: true } } },",
    derivative_include,
)
replace_once(
    service,
    "    const asset = await this.requireAsset(userId, assetId);\n    await this.storage.deleteFile(asset.storageKey);\n    await this.prisma.mediaAsset.delete({ where: { id: asset.id } });",
    """    const asset = await this.requireAsset(userId, assetId);
    const derivatives = await this.prisma.mediaDerivative.findMany({
      where: { assetId: asset.id },
      select: { storageKey: true, status: true },
    });
    for (const derivative of derivatives) {
      if (derivative.status === 'READY' && !derivative.storageKey.startsWith('pending/')) {
        await this.storage.deleteFile(derivative.storageKey).catch(() => undefined);
      }
    }
    await this.storage.deleteFile(asset.storageKey);
    await this.prisma.mediaAsset.delete({ where: { id: asset.id } });""",
)
replace_once(
    service,
    "    const tags = this.parseTags(asset.tags);\n    const smart = this.toSmartAsset(asset);",
    """    const derivativeEntries = await Promise.all(
      (asset.derivatives ?? []).map(async (derivative) => [
        derivative.kind,
        await this.storage.getSignedUrl(derivative.storageKey, 900).catch(() => null),
      ] as const),
    );
    const derivativeUrls = Object.fromEntries(derivativeEntries) as Record<string, string | null>;
    const tags = this.parseTags(asset.tags);
    const smart = this.toSmartAsset(asset);""",
)
replace_once(
    service,
    "      url,\n      urlExpiresInSeconds: url ? 900 : null,\n      status: asset.status,",
    """      url,
      thumbnailUrl: derivativeUrls.THUMBNAIL ?? null,
      posterUrl: derivativeUrls.POSTER ?? null,
      previewUrl: derivativeUrls.PREVIEW ?? null,
      urlExpiresInSeconds: url ? 900 : null,
      status: asset.status,""",
)
replace_once(
    service,
    "      include: includeAlbums ? { albumLinks: { select: { albumId: true } } } : undefined,",
    """      include: includeAlbums
        ? {
            albumLinks: { select: { albumId: true } },
            derivatives: {
              where: { status: 'READY' },
              select: { kind: true, storageKey: true, status: true },
            },
          }
        : undefined,""",
)

replace_once(
    "apps/api/src/media-library/media-library.types.ts",
    "  status: 'PENDING' | 'READY' | 'FAILED' | 'DELETED';",
    "  status: 'PENDING' | 'READY' | 'FAILED' | 'QUARANTINED' | 'DELETED';",
)

replace_once(
    "apps/web/src/lib/api/media-library.ts",
    "  url: string | null;\n  urlExpiresInSeconds: number | null;",
    "  url: string | null;\n  thumbnailUrl?: string | null;\n  posterUrl?: string | null;\n  previewUrl?: string | null;\n  urlExpiresInSeconds: number | null;",
)

# Keep legacy session consumers synchronized with the canonical auth store so
# Coach/Health/Library hydrate reliably after both login and page reload.
web_api = "apps/web/src/lib/api.ts"
replace_once(
    web_api,
    "import { useAuthStore } from './stores/auth-store';",
    "import { useAuthStore } from './stores/auth-store';\nimport { useSessionStore } from '../store/session';",
)
replace_all(
    web_api,
    "      useAuthStore.getState().setAuth(response.user, response.access_token);",
    "      useAuthStore.getState().setAuth(response.user, response.access_token);\n      useSessionStore.getState().setSession(response.user, response.access_token);",
)
replace_once(
    web_api,
    "  logout: () => {\n    useAuthStore.getState().logout();\n  },",
    "  logout: () => {\n    useAuthStore.getState().logout();\n    useSessionStore.getState().clearSession();\n  },",
)
replace_once(
    web_api,
    "  me: () => apiClient.get('/auth/me'),",
    """  me: async () => {
    const user = await apiClient.get('/auth/me');
    const token = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;
    if (token) useSessionStore.getState().setSession(user, token);
    return user;
  },""",
)

# Explicit consent to keep original media only after analysis.
behavior = "apps/web/src/app/coach/observe/page.tsx"
replace_once(
    behavior,
    "import { BottomNav } from '@/components/bottom-nav';",
    "import { BottomNav } from '@/components/bottom-nav';\nimport { KeepMediaButton } from '@/components/keep-media-button';",
)
replace_once(
    behavior,
    "            {result.observationId && (\n              <div className=\"border-t border-border/60 pt-4\">",
    """            {media && (
              <KeepMediaButton
                petId={petId}
                media={media}
                source="behavior-vision"
                linkedObservationId={result.observationId}
                tags={['behavior', context, phase]}
              />
            )}

            {result.observationId && (
              <div className="border-t border-border/60 pt-4">""",
)

health = "apps/web/src/app/health/page.tsx"
replace_once(
    health,
    "import { BottomNav } from '@/components/bottom-nav';",
    "import { BottomNav } from '@/components/bottom-nav';\nimport { KeepMediaButton } from '@/components/keep-media-button';",
)
replace_once(
    health,
    "            <AssessmentDetails assessment={result.assessment} />\n            {result.assessment.questions.length > 0 && (",
    """            <AssessmentDetails assessment={result.assessment} />
            {image && result.assessment.triage !== 'emergency_now' && (
              <div className="mt-4">
                <KeepMediaButton
                  petId={petId}
                  media={image}
                  source="health-lens"
                  linkedObservationId={result.assessmentId}
                  tags={['health', bodyArea]}
                />
              </div>
            )}
            {result.assessment.questions.length > 0 && (""",
)

library = "apps/web/src/app/library/page.tsx"
replace_once(
    library,
    '<div className="min-h-screen bg-background pb-28">',
    '<div data-testid="media-library-root" className="min-h-screen bg-background pb-28">',
)
replace_once(
    library,
    """        {notice && (
          <div role="status" className="rounded-2xl border border-primary/15 bg-primary/[0.055] px-4 py-3 text-sm">
            {notice}
          </div>
        )}""",
    """        {notice && (
          <div role="status" className="rounded-2xl border border-primary/15 bg-primary/[0.055] px-4 py-3 text-sm">
            {notice}
          </div>
        )}

        {uploadMutation.isPending && (
          <div
            data-testid="media-library-uploading"
            role="status"
            aria-live="polite"
            className="flex items-center gap-2 rounded-2xl border border-primary/15 bg-primary/[0.04] px-4 py-3 text-sm"
          >
            <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
            Saving selected originals privately…
          </div>
        )}

        {uploadMutation.isError && (
          <div role="alert" className="rounded-2xl border border-destructive/25 bg-destructive/[0.05] px-4 py-3 text-sm">
            {uploadMutation.error instanceof Error
              ? uploadMutation.error.message
              : 'One or more originals could not be saved.'}
          </div>
        )}""",
)
replace_once(
    library,
    '<section>\n          <div className="mb-3 flex items-center justify-between gap-3">',
    '<section data-testid="media-library-albums">\n          <div className="mb-3 flex items-center justify-between gap-3">',
)
replace_once(
    library,
    """            {selectedIds.length > 0 && (
              <div className="flex flex-wrap justify-end gap-2">""",
    """            {selectedIds.length > 0 && (
              <div className="flex flex-wrap items-center justify-end gap-2">
                <span className="text-xs font-semibold text-muted-foreground">
                  {selectedIds.length} selected
                </span>""",
)
replace_once(
    library,
    "        {libraryQuery.isLoading ? (",
    """        {libraryQuery.isError ? (
          <section
            data-testid="media-library-error"
            className="rounded-3xl border border-destructive/20 bg-destructive/[0.04] p-8 text-center"
          >
            <h2 className="font-semibold">Couldn’t open your private library</h2>
            <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
              Your originals are still private. Check your connection and try loading the library again.
            </p>
            <Button type="button" variant="outline" className="mt-4" onClick={() => void libraryQuery.refetch()}>
              Try again
            </Button>
          </section>
        ) : libraryQuery.isLoading ? (""",
)
replace_once(
    library,
    '<section className="rounded-3xl border border-dashed border-border p-10 text-center">',
    '<section data-testid="media-library-empty" className="rounded-3xl border border-dashed border-border p-10 text-center">',
)
replace_once(
    library,
    '<section className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">',
    '<section data-testid="media-library-grid" className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">',
)
replace_once(
    library,
    "                    aria-label={selected ? `Deselect ${asset.filename}` : `Select ${asset.filename}`}\n                    className={cn(",
    "                    aria-label={selected ? `Deselect ${asset.filename}` : `Select ${asset.filename}`}\n                    data-asset-id={asset.id}\n                    className={cn(",
)
replace_once(
    library,
    "                    {asset.mediaType === 'video' && asset.url ? (\n                      <video\n                        src={asset.url}",
    "                    {asset.mediaType === 'video' && (asset.previewUrl || asset.url) ? (\n                      <video\n                        src={asset.previewUrl ?? asset.url ?? undefined}",
)
replace_once(
    library,
    "                        style={asset.url ? { backgroundImage: `url(${JSON.stringify(asset.url).slice(1, -1)})` } : undefined}",
    "                        style={(asset.thumbnailUrl ?? asset.posterUrl ?? asset.url) ? { backgroundImage: `url(${JSON.stringify(asset.thumbnailUrl ?? asset.posterUrl ?? asset.url).slice(1, -1)})` } : undefined}",
)

# Selected-only Apple picker wording and no implicit Android microphone permission.
app_path = Path('apps/mobile/app.json')
app = json.loads(app_path.read_text())
app['expo']['ios']['infoPlist']['NSPhotoLibraryUsageDescription'] = (
    'Woof accesses only photos and videos you explicitly choose for your private pet library.'
)
for plugin in app['expo'].get('plugins', []):
    if isinstance(plugin, list) and plugin and plugin[0] == 'expo-image-picker':
        plugin[1]['photosPermission'] = (
            'Allow Woof to use only photos and videos you explicitly select.'
        )
        plugin[1]['microphonePermission'] = False
app_path.write_text(json.dumps(app, indent=2) + '\n')

# Remove the earlier duplicate component now that the shared post-analysis control is canonical.
legacy_keep = Path('apps/web/src/components/media-library/keep-media-button.tsx')
if legacy_keep.exists():
    legacy_keep.unlink()
