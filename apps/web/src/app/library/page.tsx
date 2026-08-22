'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Check,
  Download,
  FolderPlus,
  Heart,
  ImagePlus,
  Images,
  Loader2,
  MoreHorizontal,
  Share2,
  Trash2,
  Upload,
} from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { type MediaAlbum, type MediaAsset, mediaLibraryApi } from '@/lib/api/media-library';
import { requestGooglePhotosToken } from '@/lib/google-photos-oauth';
import { useAuthStore } from '@/lib/stores/auth-store';
import { cn } from '@/lib/utils';

function bytesLabel(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

async function shareOrDownload(asset: MediaAsset) {
  if (!asset.url) return;
  const response = await fetch(asset.url);
  if (!response.ok) throw new Error('Could not retrieve the media file.');
  const blob = await response.blob();
  const file = new File([blob], asset.filename, { type: asset.mimeType });
  if (navigator.share && navigator.canShare?.({ files: [file] })) {
    await navigator.share({ files: [file], title: asset.filename });
    return;
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = asset.filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function parsePollingSeconds(value?: string) {
  const parsed = Number.parseFloat(value ?? '2.5');
  return Number.isFinite(parsed) ? Math.max(1, Math.min(10, parsed)) : 2.5;
}

export default function MediaLibraryPage() {
  const user = useAuthStore((state) => state.user);
  const pets = user?.pets ?? [];
  const [petId, setPetId] = useState(pets[0]?.id ?? '');
  const [albumId, setAlbumId] = useState<string | undefined>();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [newAlbumOpen, setNewAlbumOpen] = useState(false);
  const [newAlbumName, setNewAlbumName] = useState('');
  const [newAlbumDescription, setNewAlbumDescription] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const queryClient = useQueryClient();

  const libraryQuery = useQuery({
    queryKey: ['media-library', petId, albumId],
    queryFn: () => mediaLibraryApi.library(petId, { albumId, limit: 80 }),
    enabled: Boolean(petId),
    retry: 1,
  });

  const refresh = () => queryClient.invalidateQueries({ queryKey: ['media-library', petId] });
  const currentCustomAlbum = libraryQuery.data?.albums.find(
    (album) => album.id === albumId && album.kind === 'USER'
  );

  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      const uploaded: MediaAsset[] = [];
      for (const file of files.slice(0, 20)) {
        uploaded.push(
          await mediaLibraryApi.uploadMedia({
            petId,
            media: file,
            filename: file.name,
            source: 'device-picker',
            albumIds: currentCustomAlbum ? [currentCustomAlbum.id] : [],
          })
        );
      }
      return uploaded;
    },
    onSuccess: (assets) => {
      setNotice(`${assets.length} ${assets.length === 1 ? 'moment' : 'moments'} added privately.`);
      void refresh();
    },
  });

  const createAlbumMutation = useMutation({
    mutationFn: () => mediaLibraryApi.createAlbum(petId, newAlbumName, newAlbumDescription),
    onSuccess: (album) => {
      setNewAlbumOpen(false);
      setNewAlbumName('');
      setNewAlbumDescription('');
      setAlbumId(album.id);
      void refresh();
    },
  });

  const favoriteMutation = useMutation({
    mutationFn: (asset: MediaAsset) =>
      mediaLibraryApi.updateAsset(asset.id, { favorite: !asset.favorite }),
    onSuccess: () => void refresh(),
  });

  const deleteMutation = useMutation({
    mutationFn: (assetId: string) => mediaLibraryApi.deleteAsset(assetId),
    onSuccess: (_, assetId) => {
      setSelectedIds((current) => current.filter((id) => id !== assetId));
      if (detailId === assetId) setDetailId(null);
      void refresh();
    },
  });

  const googleImportMutation = useMutation({
    mutationFn: async () => {
      const accessToken = await requestGooglePhotosToken('import');
      const session = await mediaLibraryApi.startGooglePhotosPicker(petId, accessToken, 50);
      const picker = window.open(
        session.pickerUri,
        'woof-google-photos',
        'popup,width=560,height=760,noopener,noreferrer'
      );
      if (!picker) throw new Error('Allow the Google Photos picker popup to continue.');

      const intervalSeconds = parsePollingSeconds(session.pollingConfig?.pollInterval);
      const startedAt = Date.now();
      while (Date.now() - startedAt < 3 * 60 * 1000) {
        await new Promise((resolve) => window.setTimeout(resolve, intervalSeconds * 1000));
        const result = await mediaLibraryApi.importGooglePhotos(
          petId,
          accessToken,
          session.sessionId,
          currentCustomAlbum ? [currentCustomAlbum.id] : []
        );
        if (result.ready) return result.imported;
      }
      throw new Error('Google Photos selection timed out. Try the import again.');
    },
    onSuccess: (assets) => {
      setNotice(
        `${assets.length} Google Photos ${assets.length === 1 ? 'item' : 'items'} imported.`
      );
      void refresh();
    },
  });

  const googleExportMutation = useMutation({
    mutationFn: async () => {
      const accessToken = await requestGooglePhotosToken('export');
      return mediaLibraryApi.exportGooglePhotos(petId, accessToken, selectedIds);
    },
    onSuccess: (result) =>
      setNotice(
        `${result.uploaded} selected ${result.uploaded === 1 ? 'item was' : 'items were'} exported to Google Photos.`
      ),
  });

  const exportManifestMutation = useMutation({
    mutationFn: () =>
      mediaLibraryApi.exportManifest(petId, selectedIds.length ? selectedIds : undefined),
    onSuccess: (manifest) => {
      const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `woof-media-${petId}-${new Date().toISOString().slice(0, 10)}.json`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  });

  const assets = libraryQuery.data?.assets ?? [];
  const albums = libraryQuery.data?.albums ?? [];
  const detailAsset = assets.find((asset) => asset.id === detailId) ?? null;
  const usagePercent = libraryQuery.data?.storage.quotaBytes
    ? Math.min(
        100,
        (libraryQuery.data.storage.usedBytes / libraryQuery.data.storage.quotaBytes) * 100
      )
    : 0;
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);

  return (
    <div className="min-h-screen bg-background pb-28">
      <header className="sticky top-0 z-30 border-b border-border/50 bg-background/90 backdrop-blur-xl">
        <div className="mx-auto max-w-4xl px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="eyebrow">Private pet library</p>
              <h1 className="mt-1 text-2xl font-bold tracking-tight">Moments that teach Woof</h1>
            </div>
            <Button type="button" size="sm" onClick={() => fileInputRef.current?.click()}>
              <ImagePlus className="mr-2 h-4 w-4" aria-hidden="true" />
              Add
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl space-y-6 px-4 py-5">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          multiple
          className="hidden"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length) uploadMutation.mutate(files);
            event.currentTarget.value = '';
          }}
        />

        {pets.length > 1 && (
          <select
            value={petId}
            onChange={(event) => {
              setPetId(event.target.value);
              setAlbumId(undefined);
              setSelectedIds([]);
            }}
            className="h-11 w-full rounded-xl border border-border bg-card px-3 text-sm sm:w-64"
            aria-label="Pet"
          >
            {pets.map((pet) => (
              <option key={pet.id} value={pet.id}>
                {pet.name}
              </option>
            ))}
          </select>
        )}

        <section className="rounded-3xl border border-border/60 bg-card/70 p-4 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold">Bring your pet history together</p>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
                Choose photos or videos from this device, the iPhone/iPad system picker, or Google
                Photos. Woof stores only what you explicitly keep here.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => googleImportMutation.mutate()}
                disabled={!petId || googleImportMutation.isPending}
              >
                {googleImportMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Upload className="mr-2 h-4 w-4" aria-hidden="true" />
                )}
                Google Photos
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setNewAlbumOpen(true)}
              >
                <FolderPlus className="mr-2 h-4 w-4" aria-hidden="true" />
                Album
              </Button>
            </div>
          </div>

          {libraryQuery.data && (
            <div className="mt-5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{bytesLabel(libraryQuery.data.storage.usedBytes)} stored</span>
                <span>{bytesLabel(libraryQuery.data.storage.quotaBytes)} private quota</span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${usagePercent}%` }}
                />
              </div>
            </div>
          )}
        </section>

        {notice && (
          <div
            role="status"
            className="rounded-2xl border border-primary/15 bg-primary/[0.055] px-4 py-3 text-sm"
          >
            {notice}
          </div>
        )}

        {uploadMutation.isPending && (
          <div
            data-testid="media-library-uploading"
            role="status"
            className="flex items-center rounded-2xl border border-primary/15 bg-primary/[0.055] px-4 py-3 text-sm"
          >
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
            <span>Saving your moment privately…</span>
          </div>
        )}

        {newAlbumOpen && (
          <section className="rounded-3xl border border-border/70 bg-card p-5">
            <h2 className="font-semibold">New album</h2>
            <div className="mt-4 grid gap-3">
              <input
                value={newAlbumName}
                onChange={(event) => setNewAlbumName(event.target.value)}
                placeholder="Weekend adventures"
                className="h-11 rounded-xl border border-border bg-background px-3 text-sm"
                maxLength={80}
              />
              <input
                value={newAlbumDescription}
                onChange={(event) => setNewAlbumDescription(event.target.value)}
                placeholder="Optional description"
                className="h-11 rounded-xl border border-border bg-background px-3 text-sm"
                maxLength={280}
              />
              <div className="flex justify-end gap-2">
                <Button type="button" variant="ghost" onClick={() => setNewAlbumOpen(false)}>
                  Cancel
                </Button>
                <Button
                  type="button"
                  disabled={!newAlbumName.trim() || createAlbumMutation.isPending}
                  onClick={() => createAlbumMutation.mutate()}
                >
                  Create album
                </Button>
              </div>
            </div>
          </section>
        )}

        <section data-testid="media-library-albums">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="font-semibold">Albums</h2>
            {selectedIds.length > 0 && (
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => googleExportMutation.mutate()}
                  disabled={googleExportMutation.isPending}
                >
                  Export {selectedIds.length} to Google
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={() => exportManifestMutation.mutate()}
                >
                  <Download className="mr-1.5 h-4 w-4" aria-hidden="true" />
                  Manifest
                </Button>
              </div>
            )}
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            <button
              type="button"
              onClick={() => setAlbumId(undefined)}
              className={cn(
                'shrink-0 rounded-full border px-3 py-2 text-xs font-semibold',
                !albumId ? 'border-primary bg-primary/10 text-primary' : 'border-border/70 bg-card'
              )}
            >
              All
            </button>
            {albums.map((album) => (
              <button
                key={album.id}
                type="button"
                onClick={() => setAlbumId(album.id)}
                className={cn(
                  'shrink-0 rounded-full border px-3 py-2 text-xs font-semibold',
                  albumId === album.id
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border/70 bg-card text-muted-foreground'
                )}
              >
                {album.name} · {album.count}
              </button>
            ))}
          </div>
        </section>

        {libraryQuery.isLoading ? (
          <div className="flex min-h-48 items-center justify-center text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
            <span className="ml-2 text-sm">Opening the private library…</span>
          </div>
        ) : libraryQuery.isError ? (
          <section
            data-testid="media-library-error"
            className="rounded-3xl border border-destructive/25 bg-destructive/[0.045] p-6"
          >
            <h2 className="font-semibold">Private library unavailable</h2>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              Woof could not open this private library right now. Your media has not been changed.
            </p>
            <Button
              type="button"
              variant="outline"
              className="mt-4"
              onClick={() => void libraryQuery.refetch()}
            >
              Try again
            </Button>
          </section>
        ) : assets.length === 0 ? (
          <section
            data-testid="media-library-empty"
            className="rounded-3xl border border-dashed border-border p-10 text-center"
          >
            <Images className="mx-auto h-8 w-8 text-muted-foreground" aria-hidden="true" />
            <h2 className="mt-3 font-semibold">No moments here yet</h2>
            <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
              Keep a behavior clip, health photo, favorite adventure, or import a few moments from
              your existing photo library.
            </p>
          </section>
        ) : (
          <section
            data-testid="media-library-grid"
            className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4"
          >
            {assets.map((asset) => {
              const selected = selectedSet.has(asset.id);
              return (
                <article
                  key={asset.id}
                  data-asset-id={asset.id}
                  className="group relative overflow-hidden rounded-2xl bg-muted"
                >
                  <button
                    type="button"
                    className="block aspect-square w-full overflow-hidden"
                    aria-label={`Open ${asset.filename}`}
                    onClick={() => setDetailId(asset.id)}
                  >
                    {asset.mediaType === 'video' && asset.url ? (
                      <video
                        src={asset.url}
                        className="h-full w-full object-cover"
                        muted
                        playsInline
                        preload="metadata"
                      />
                    ) : (
                      <div
                        className="h-full w-full bg-cover bg-center transition-transform duration-300 group-hover:scale-[1.02]"
                        style={
                          asset.url
                            ? { backgroundImage: `url(${JSON.stringify(asset.url).slice(1, -1)})` }
                            : undefined
                        }
                      />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setSelectedIds((current) =>
                        current.includes(asset.id)
                          ? current.filter((id) => id !== asset.id)
                          : [...current, asset.id]
                      )
                    }
                    aria-label={
                      selected ? `Deselect ${asset.filename}` : `Select ${asset.filename}`
                    }
                    className={cn(
                      'absolute left-2 top-2 flex h-8 w-8 items-center justify-center rounded-full border backdrop-blur-md',
                      selected
                        ? 'border-primary bg-primary text-primary-foreground'
                        : 'border-white/30 bg-black/30 text-white'
                    )}
                  >
                    {selected ? <Check className="h-4 w-4" aria-hidden="true" /> : null}
                  </button>
                  {asset.favorite && (
                    <Heart
                      className="absolute bottom-2 right-2 h-4 w-4 fill-current text-white drop-shadow"
                      aria-hidden="true"
                    />
                  )}
                </article>
              );
            })}
          </section>
        )}

        {detailAsset && (
          <section className="rounded-3xl border border-border/70 bg-card p-5">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate font-semibold">{detailAsset.filename}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {detailAsset.source} · {bytesLabel(detailAsset.sizeBytes)}
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setDetailId(null)}
                aria-label="Close details"
              >
                <MoreHorizontal className="h-5 w-5" aria-hidden="true" />
              </Button>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {detailAsset.tags.map((tag) => (
                <span
                  key={`${tag.source}-${tag.label}`}
                  className="rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground"
                >
                  {tag.label}
                </span>
              ))}
            </div>

            <div className="mt-5 grid gap-2 sm:grid-cols-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => favoriteMutation.mutate(detailAsset)}
              >
                <Heart
                  className={cn('mr-2 h-4 w-4', detailAsset.favorite && 'fill-current')}
                  aria-hidden="true"
                />
                {detailAsset.favorite ? 'Favorited' : 'Favorite'}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() =>
                  void shareOrDownload(detailAsset).catch((error) =>
                    setNotice(error instanceof Error ? error.message : 'Could not share this item.')
                  )
                }
              >
                <Share2 className="mr-2 h-4 w-4" aria-hidden="true" />
                Share / save
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => deleteMutation.mutate(detailAsset.id)}
              >
                <Trash2 className="mr-2 h-4 w-4" aria-hidden="true" />
                Delete
              </Button>
            </div>

            {albums.some((album) => album.kind === 'USER') && (
              <div className="mt-5">
                <p className="text-sm font-semibold">Custom albums</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {albums
                    .filter((album): album is MediaAlbum => album.kind === 'USER')
                    .map((album) => {
                      const inAlbum = detailAsset.albumIds.includes(album.id);
                      return (
                        <button
                          key={album.id}
                          type="button"
                          onClick={() => {
                            const next = inAlbum
                              ? detailAsset.albumIds.filter((id) => id !== album.id)
                              : [...detailAsset.albumIds, album.id];
                            mediaLibraryApi
                              .updateAsset(detailAsset.id, { albumIds: next })
                              .then(() => refresh())
                              .catch(() => setNotice('Could not update this album.'));
                          }}
                          className={cn(
                            'rounded-full border px-3 py-2 text-xs font-semibold',
                            inAlbum
                              ? 'border-primary bg-primary/10 text-primary'
                              : 'border-border/70 text-muted-foreground'
                          )}
                        >
                          {album.name}
                        </button>
                      );
                    })}
                </div>
              </div>
            )}
          </section>
        )}

        <section className="rounded-3xl border border-border/60 p-5">
          <p className="text-sm font-semibold">Portability first</p>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Woof uses system-selected imports rather than broad photo-library access. Originals can
            be shared or downloaded at any time, exported back to Google Photos, or represented by a
            portable JSON manifest with short-lived download links.
          </p>
        </section>
      </main>
      <BottomNav />
    </div>
  );
}
